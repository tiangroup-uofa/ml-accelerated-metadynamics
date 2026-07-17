#!/usr/bin/env python3
"""
extract_epot.py
===============
Parse the molecular-dynamics potential energy (Epot) from the TABULAR MD
section that xTB 6.7.1 writes to ``output.log``.

xTB 6.7.1 prints the MD progress as a numeric table whose header names
the columns, e.g.::

         step   time(ps)    Epot       Ekin      <T>      T     Etot
          200    0.20    -10.09118   0.00456  ...
          400    0.40    -10.11436   0.00461  ...

The exact spacing and the precise set of columns can vary slightly, so
this parser:

  * locates the header line by the presence of the words "time", "Epot"
    and "Ekin" (case-insensitive);
  * maps column positions from that header;
  * then parses only the numeric rows that follow, stopping when the
    table ends (a non-numeric / blank line);
  * supports decimal, ``E`` scientific and Fortran ``D`` notation;
  * drops an invalid initialization row whose Epot is a zero placeholder;
  * de-duplicates repeated (step, time) rows keeping the LAST valid one.

Before relying on it, inspect the file yourself::

    grep -n "Epot" output.log
    grep -n "time" output.log

Output
------
    epot.csv  with columns:
        step, time_ps, epot_hartree,
        epot_relative_hartree, epot_relative_ev, epot_relative_kj_mol

Energies are reported RELATIVE to the minimum sampled Epot.

Usage
-----
    python extract_epot.py --log ../run/output.log --out ../analysis/epot.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HARTREE_TO_EV = 27.211386
HARTREE_TO_KJMOL = 2625.4996

# Float allowing decimal, E-scientific and Fortran D-scientific notation.
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
FLOAT_RE = re.compile(FLOAT)

# Header detection: a line that simultaneously mentions time, Epot, Ekin.
HEADER_TOKENS = ("time", "epot", "ekin")


def to_float(token: str) -> float:
    """Convert a numeric token, handling Fortran 'D'/'d' exponents."""
    return float(token.replace("D", "E").replace("d", "e"))


def find_header(lines: list[str]) -> tuple[int, list[str]] | None:
    """Return (line_index, lowercased_tokens) of the MD table header.

    The header is the first line containing all of 'time', 'Epot', 'Ekin'.
    """
    for i, line in enumerate(lines):
        low = line.lower()
        if all(tok in low for tok in HEADER_TOKENS):
            # Whitespace-separated column labels, one per data column.
            # Drop tokens that canonicalise to empty: these are detached
            # unit annotations like "(ps)" that belong to the preceding
            # label and are NOT separate data columns. Keeping them would
            # over-count the header width and break column alignment.
            labels = [_canonical_label(t) for t in line.split()]
            labels = [lab for lab in labels if lab]
            return i, labels
    return None


def _canonical_label(token: str) -> str:
    """Reduce a header label to a bare canonical name.

    Strips parenthetical units and angle brackets so that e.g.
    'time(ps)' -> 'time', '<T>' -> 't', 'Epot' -> 'epot'. Each header
    label corresponds to exactly one numeric data column, so position is
    preserved 1:1 with the numbers in a data row.
    """
    low = token.lower()
    low = re.sub(r"\(.*?\)", "", low)        # drop "(ps)" etc.
    low = low.strip("<>")                     # "<t>" -> "t"
    low = re.sub(r"[^a-z]", "", low)          # keep letters only
    return low


def header_column_map(tokens: list[str]) -> dict[str, int]:
    """Map step/time/Epot to their column index among the data numbers.

    Header labels were canonicalised by _canonical_label, and each header
    label lines up 1:1 with a numeric field in the data rows, so the
    header index is directly the numeric-column index. xTB headers
    typically read: step time(ps) Epot Ekin <T> T Etot. Some builds omit
    an explicit 'step' column; the caller then falls back to a counter.
    """
    col = {}
    for idx, name in enumerate(tokens):
        if name == "step" and "step" not in col:
            col["step"] = idx
        elif name == "time" and "time" not in col:
            col["time"] = idx
        elif name == "epot" and "epot" not in col:
            col["epot"] = idx
    return col


def parse_log(text: str) -> pd.DataFrame:
    """Parse the xTB MD energy table into step/time/Epot records.

    Verified against real xTB 6.7.1 output, whose MD table header is::

        time (ps)    <Epot>      Ekin   <T>   T     Etot

    Two important real-format facts handled here:

    * The DATA rows carry a leading MD-step integer that is NOT named in
      the header (header has N labels, data rows have N+1 numbers). We
      detect that offset and shift the column indices accordingly, so the
      unlabelled step column is consumed correctly and 'time' / '<Epot>'
      stay aligned.
    * Status lines such as "adding snapshot to metadynamics bias" and
      "est. speed in wall clock ..." appear INTERLEAVED inside the table.
      Those are skipped, not treated as the end of the table, so the full
      run is parsed.
    """
    lines = text.splitlines()
    found = find_header(lines)
    if found is None:
        return pd.DataFrame(columns=["step", "time_ps", "epot_hartree"])

    header_idx, tokens = found
    colmap = header_column_map(tokens)
    if "time" not in colmap or "epot" not in colmap:
        # header matched the keywords but columns are unexpected
        return pd.DataFrame(columns=["step", "time_ps", "epot_hartree"])

    n_labels = len(tokens)
    # A genuine data row consists ONLY of numbers (incl. E/D exponents)
    # and whitespace; anything with other letters is a status/prose line.
    data_row_re = re.compile(r"^[\s\d.+\-eEdD]+$")

    # Determine the leading-column offset from the first real data row:
    # if the row has one more number than the header has labels, there is
    # an unlabelled leading step column (offset = 1).
    offset = None
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped or not data_row_re.match(stripped):
            continue
        nums = FLOAT_RE.findall(line)
        if len(nums) >= n_labels:
            offset = len(nums) - n_labels
            break
    if offset is None:
        return pd.DataFrame(columns=["step", "time_ps", "epot_hartree"])

    time_idx = colmap["time"] + offset
    epot_idx = colmap["epot"] + offset
    needed = max(time_idx, epot_idx) + 1

    records = []
    auto_step = 0

    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Interleaved status / prose lines (letters present) -> skip, but
        # KEEP scanning: the table resumes on the next numeric row.
        if not data_row_re.match(stripped):
            continue

        nums = FLOAT_RE.findall(line)
        # Require the exact table width (n_labels + offset). This rejects
        # the later stray "Epot : -2.58..." summary lines, which have far
        # fewer numbers than a full MD table row.
        if len(nums) != n_labels + offset:
            continue

        try:
            time_ps = to_float(nums[time_idx])
            epot = to_float(nums[epot_idx])
        except (ValueError, IndexError):
            continue

        # Leading step column when present (offset covers it); otherwise
        # use a running counter.
        if offset >= 1:
            try:
                step = int(round(to_float(nums[0])))
            except (ValueError, IndexError):
                step = auto_step
        else:
            step = auto_step
        auto_step += 1

        # Exclude the t=0 initialization row whose <Epot> is a 0.0
        # placeholder rather than a real energy.
        if epot == 0.0:
            continue

        records.append((step, time_ps, epot))

    df = pd.DataFrame(records, columns=["step", "time_ps", "epot_hartree"])
    if df.empty:
        return df

    # Remove duplicate (step, time) rows, keeping the LAST valid record.
    df = df.drop_duplicates(subset=["step", "time_ps"], keep="last")
    df = df.sort_values("time_ps").reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default paths assume execution from xtb_metadynamics/scripts/.
    parser.add_argument("--log", default="../run/output.log", type=Path)
    parser.add_argument("--out", default="../analysis/epot.csv", type=Path)
    args = parser.parse_args()

    log_path = args.log.resolve()
    if not log_path.exists():
        print(f"ERROR: log file {log_path} not found.", file=sys.stderr)
        return 1

    text = log_path.read_text(errors="replace")
    df = parse_log(text)

    if df.empty:
        print(
            "ERROR: no valid MD energy table was found in the log.\n"
            "xTB 6.7.1 should print a table whose header contains "
            "'time', 'Epot' and 'Ekin'. Inspect the file manually:\n"
            f"    grep -n 'Epot' {log_path}\n"
            f"    grep -n 'time' {log_path}\n"
            "Then adjust HEADER_TOKENS / header_column_map() in this "
            "script to match your build's column layout.",
            file=sys.stderr,
        )
        return 2

    emin = df["epot_hartree"].min()
    df["epot_relative_hartree"] = df["epot_hartree"] - emin
    df["epot_relative_ev"] = df["epot_relative_hartree"] * HARTREE_TO_EV
    df["epot_relative_kj_mol"] = df["epot_relative_hartree"] * HARTREE_TO_KJMOL

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Parsed {len(df)} MD energy rows -> {args.out.resolve()}")
    print(f"  time range:           {df['time_ps'].min():.3f} .. "
          f"{df['time_ps'].max():.3f} ps")
    print(f"  Epot min (reference): {emin:.8f} Hartree")
    print(f"  max relative energy:  {df['epot_relative_kj_mol'].max():.2f} kJ/mol")
    return 0


if __name__ == "__main__":
    sys.exit(main())

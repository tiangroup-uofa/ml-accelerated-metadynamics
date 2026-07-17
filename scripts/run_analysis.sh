#!/bin/bash
# =====================================================================
# run_analysis.sh -- run the full post-processing pipeline in order.
#
# Assumes the xTB metadynamics run has already finished and produced
# scoord.* and output.log in the RUN_DIR.
#
# Usage (from the scripts/ directory):
#     ./run_analysis.sh
#
# Configurable via environment variables:
#     RUN_DIR             dir with scoord.* and output.log  (../run)
#     ANALYSIS_DIR        dir for CSV / trajectory outputs   (../analysis)
#     FIG_DIR             dir for PNG figures                (../figures)
#     PYTHON              python interpreter                 (python3)
#     DT_PS               approx ps between snapshots        (1.0)
#     T0_PS              approx time of first snapshot (ps)  (1.0)
#     TOLERANCE_PS        energy-match tolerance (ps)        (0.5)
#     CONNECTIVITY_SCALE  covalent-radii scale for fragments (1.3)
#
# Example:
#     DT_PS=1.0 TOLERANCE_PS=0.2 CONNECTIVITY_SCALE=1.25 ./run_analysis.sh
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/../run}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${SCRIPT_DIR}/../analysis}"
FIG_DIR="${FIG_DIR:-${SCRIPT_DIR}/../figures}"
PY="${PYTHON:-python3}"

DT_PS="${DT_PS:-1.0}"
T0_PS="${T0_PS:-1.0}"
TOLERANCE_PS="${TOLERANCE_PS:-0.5}"
CONNECTIVITY_SCALE="${CONNECTIVITY_SCALE:-1.3}"

require() {
    if [[ ! -e "$1" ]]; then
        echo "ERROR: required input not found: $1" >&2
        exit 1
    fi
}

echo "==> [1/5] Converting scoord.* structures"
require "${RUN_DIR}"
"${PY}" "${SCRIPT_DIR}/convert_scoord.py" \
    --run-dir "${RUN_DIR}" --out-dir "${ANALYSIS_DIR}"

echo "==> [2/5] Calculating collective variables"
require "${ANALYSIS_DIR}/scoord_1ps.xyz"
"${PY}" "${SCRIPT_DIR}/calculate_cvs.py" \
    --xyz "${ANALYSIS_DIR}/scoord_1ps.xyz" \
    --out "${ANALYSIS_DIR}/collective_variables.csv" \
    --dt-ps "${DT_PS}" \
    --t0-ps "${T0_PS}" \
    --connectivity-scale "${CONNECTIVITY_SCALE}"

echo "==> [3/5] Extracting Epot from the xTB log"
require "${RUN_DIR}/output.log"
"${PY}" "${SCRIPT_DIR}/extract_epot.py" \
    --log "${RUN_DIR}/output.log" \
    --out "${ANALYSIS_DIR}/epot.csv"

echo "==> [4/5] Merging structures with energies"
require "${ANALYSIS_DIR}/collective_variables.csv"
require "${ANALYSIS_DIR}/epot.csv"
"${PY}" "${SCRIPT_DIR}/merge_structure_energy.py" \
    --cvs "${ANALYSIS_DIR}/collective_variables.csv" \
    --epot "${ANALYSIS_DIR}/epot.csv" \
    --out "${ANALYSIS_DIR}/trajectory_analysis.csv" \
    --tolerance-ps "${TOLERANCE_PS}"

echo "==> [5/5] Generating figures"
require "${ANALYSIS_DIR}/trajectory_analysis.csv"
"${PY}" "${SCRIPT_DIR}/plot_analysis.py" \
    --data "${ANALYSIS_DIR}/trajectory_analysis.csv" \
    --fig-dir "${FIG_DIR}"

echo "==> Pipeline complete."
echo "    CSVs:    ${ANALYSIS_DIR}"
echo "    Figures: ${FIG_DIR}"

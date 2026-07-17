# Diels-Alder in the xTB RMSD-bias setup — can we see the reaction?

**Yes.** Butadiene + ethylene → cyclohexene was driven and characterised at
GFN2-xTB. This is the same RMSD-bias machinery as the water metadynamics, and
it makes a clean methodological point about what that bias does and doesn't give.

## What was done

1. **Endpoints** (`build_diels_alder.py`): s-cis butadiene + ethylene reactant
   complex and cyclohexene product, built with **matched atom ordering** (the
   `--path` interpolation requires it), then GFN2 `--opt` → `start.xyz`, `end.xyz`.
   Verified chemically correct: reactant has two separate π-systems (forming
   C–C 3.35 Å, diene 1.33/1.45/1.33, ethylene 1.32); product is cyclohexene
   (new σ-bonds 1.54 Å, new C2=C3 1.32, former ethylene 1.55).

2. **Reaction driven** (`run_path.py`, `xtb --path` with the tutorial settings
   kpush=0.003 / kpull=−0.015 / ppull=0.05): the path connected reactant → product
   (product-end RMSD 0.001), forming **both** new C–C σ-bonds. Reaction captured.

3. **Barrier — done properly** (this took several tries; see "Lesson"):
   - `xtb --path` "forward barrier" = **101 kcal/mol → discard.** It is the max
     energy along the *biased* path, and the driver had to escalate the bias
     (run 5: kpush 0.003→0.015) to connect, forcing a high-energy over-synchronous
     geometry.
   - Relaxed **concerted** scan (`cscan/`, both forming bonds constrained to the
     same value, 3.30→1.54 Å): smooth barrier peaking at **6.7 kcal/mol at
     forming-bond distance 2.32 Å** — a textbook symmetric DA TS.
   - **Verified TS** (`refine_ts_sella.py`, Sella order-1 from the *good* scan
     guess + `xtb --hess`): converges at forming bonds **2.32 Å**, barrier
     **6.7 kcal/mol** (independent agreement with the scan), and the `vibspectrum`
     shows **exactly one imaginary mode at −394 cm⁻¹** = the symmetric
     bond-forming reaction coordinate. Genuine first-order saddle. → `ts_opt.xyz`.
   - A *sequential* scan (form one bond, then the other) gives a **stepwise**
     first-bond barrier of **19.3 kcal/mol** — higher than the concerted 6.7, so
     GFN2 prefers the concerted path, as expected for butadiene+ethylene.

## Numbers

| quantity | value | note |
|---|---|---|
| forming C–C at TS | 2.32 Å (symmetric) | textbook concerted DA |
| imaginary frequency | −394 cm⁻¹ (only one) | verified genuine TS |
| **concerted barrier** | **6.7 kcal/mol** | scan and Sella+Hessian agree |
| stepwise (1st bond) barrier | 19.3 kcal/mol | concerted is preferred |
| reaction energy | −57.6 kcal/mol | optimized endpoints, single points |
| (`--path` biased-path "barrier") | 101 kcal/mol | **not physical — do not quote** |
| (tutorial reference) | ~12.4 kcal/mol, −25 | different setup; GFN2 underbinds DA barriers |

GFN2-xTB is known to underestimate Diels-Alder barriers, consistent with
6.7 vs the tutorial's ~12; the −394 cm⁻¹ mode and 2.32 Å TS geometry are the
robust, method-independent checks that this is the right saddle.

## Lesson (ties back to the metadynamics theme)

The RMSD bias found *a* route to product but **not the minimum-energy one**, and
its "barrier" was meaningless — exactly the chemical-blindness of a global-RMSD
bias we saw evaporating the water clusters. Enhanced-sampling/path bias is great
for **discovering that a reaction is reachable and generating a rough path**, but
a **quantitative barrier needs a proper TS treatment** (relaxed scan → saddle
optimizer → Hessian). Two engineering notes worth keeping:
- xtb has **no built-in TS optimizer** (`--optts` is not a flag); use an external
  saddle optimizer (Sella) with xtb as the ASE calculator.
- Do **not** scrape frequencies from the `--hess` stdout by regex — it also
  matches the `imag. cutoff -20.0 cm⁻¹` thermo *parameter*. Read the
  `vibspectrum` file instead.

## Reproduce

```bash
export XTB_BIN=$(which xtb)
python build_diels_alder.py
python run_path.py                                   # drives the reaction, writes xtbpath_ts.xyz
python scan_ts.py                                    # sequential scan (stepwise 19.3)
# concerted scan is in cscan/ (both bonds constrained equal) -> cscan_ts_guess.xyz
micromamba run -n xtb python refine_ts_sella.py      # verified TS: 6.7 kcal/mol, -394 cm^-1
```

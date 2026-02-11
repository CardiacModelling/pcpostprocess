# Lei 2019 procedure

## Per well

1. `estimate_gE_leak_ramp` (staircase) or `estimate_gE_leak_step` (validation signals)
   - linear regression
   - **Figure S2B** in the publication (Rapid I)
2. `correct_gE_leak_ramp`
    Having separate from estimation would allow estimate to be re-used on traces without ramps or suitable steps
3. E-4031 subtraction (0.5uM E-4031 vs dimethyl vehicle)
   **Not sure where to implement**
   - **Figure S2A**
4. `estimate_reversal_potential_ramp`
   - Third order poylnomial
   - **Figure S2C** or **Figure 10A**
   - Output: EK
   - In Chon's code `fit_EK_poly` in `qc/analyse-ek.py`
5. Correct leak correction (**validation protocols only (except act/inact?**)
   - "These leak corrections can overcorrect or undercorrect."
   - "IKr should only be negative when the voltage is below its reversal potential, approximately -85.2 mV (NERNST)"
   - "If the leak-corrected current showed a negative current at voltages **substantially larger** than NERNST, we concluded we overestimated leak."
   - Most noticeable during highest V in staircase (max I leak)
   - **For each validation proto, we specified a time window during which we believe IKr should be almost zero (please refer to our GitHub repository for detail).**
   - To rectify the over- or undercorrection, we re-estimated the leak correction by adding an extra linear leak current of the form `g * (V + 80)`, with `g` chosen to make `mean(I_window) = 0`

## QC - before time series

1. `QC No cell`: Not in paper: Well not listed, or `R_seal`, `C_m`, or `R_series` is `None`
2. `QC1.Rseal`: R_seal in [.1, 1000] GOhm
   - Needs trace that provides R_seal, needs lower and upper bound
3. `QC1.Cm`: C_m in [1, 100] pF
   - Needs trace that provides C_m, needs lower and upper bound
4. `QC1.Rseries`: R_series in [1, 25] MOhm
   - Needs trace that provides R_series, needs lower and upper bound
   
## QC - before subtraction

1. `QC2.raw` as `snr = (np.std(c) / np.std(c[:200]))**2` and `snr < 25 or not np.isfinite(np.std(c))`

   
   
   
   
   
   
## Aggregate

1. EK histogram of 124 accepted wells
   - **Figure 10B**
   
   
   
   
   
   
X. Bar plots of number filtered out on each QC crit
   
   
   
   
   
## Maybe

- `estimate_max_conductance`
  In Supplement to Rapid 2, showing a single exp fit to a tail current and
  using the max as max current
the max value

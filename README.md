# Roman Bulge Calibration Catalog Pipeline

This pipeline builds a cross-matched, dereddened, spectroscopically-characterized 
calibration catalog of stars in the Roman Space Telescope Galactic Bulge Time-Domain 
Survey (GBTDS) footprint. The catalog is the observational backbone of Project B 
(Joyce, Tayar et al., in prep): it collects every independently constrained stellar 
parameter available for stars in the footprint to support model calibration and 
age-metallicity relation inference.

---

## Repo layout

```
ROMAN/
├── README.md                          ← this file
│
├── Input data  (not tracked by git)
│   ├── gaiaXGBoost.fits               8.0 GB   Gaia DR3 + XGBoost photometric parameters
│   ├── astra.fits                     2.2 GB   ASTRA DR1 (Silva et al. 2026) spectroscopic params
│   ├── bdbs.fits                      307 MB   BDBS optical photometry (Johnson et al. 2022)
│   └── bensby.fits                    830 KB   Bensby+2017 high-resolution spectroscopy sample
│
├── Pipeline scripts  (run in order)
│   ├── step1_load_and_merge.py        Merge Gaia+XGBoost and ASTRA into a single table
│   ├── step2_footprint_and_cuts.py    Apply GBTDS tile geometry and Weiss-Zinn stellar cuts
│   ├── step3_remote_xmatch.py         CDS cross-matches: VIRAC2, 2MASS, VVV reddening map
│   ├── step4_photometry.py            (optional) Re-derive best NIR photometry and extinction
│   ├── step5_parameters_and_output.py Harvest all parameters, build final clean catalog
│   ├── step6_plots.py                 Full diagnostic plot suite (reads roman_master.fits)
│   ├── step7_bdbs.py                  BDBS-derived quantities and updated plots
│   └── plotssky.py                    Publication sky-coverage map for the proposal/paper
│
├── Intermediate outputs  (not tracked by git)
│   ├── merged.fits                    2.4 GB   output of step 1
│   ├── step2_selected.fits            2.1 MB   output of step 2
│   ├── step3_xmatched.fits            2.6 MB   output of step 3
│   ├── step4_photometry.fits          2.7 MB   output of step 4 (if run)
│   ├── step3_summary.txt                       match statistics from step 3
│   └── bdbs_match.csv                 9.3 KB   BDBS cross-match table (used in TOPCAT step)
│
├── Final outputs  (not tracked by git)
│   ├── roman_master.fits              740 KB   full clean catalog (step 5)
│   ├── roman_master.csv               861 KB
│   ├── roman_calibration_ready.fits   231 KB   subset with complete calibration parameters
│   ├── roman_calibration_ready.csv    317 KB
│   ├── roman_summary.txt                       catalog statistics
│   ├── roman_master_bdbs.fits         777 KB   roman_master + BDBS quantities (step 7)
│   ├── roman_master_bdbs.csv          873 KB
│   └── plots/                                  all figures (steps 6, 7, plotssky.py)
│
└── Legacy / exploratory scripts  (kept for reference)
    ├── pipeline.py                    53 KB    original monolithic pipeline
    ├── roman_calibration_catalog.py
    ├── roman_final_calibration_catalog.py
    ├── roman_parameter_catalog.py
    ├── roman_deredden_sightlines.py   (ported into step 3)
    ├── roman_calibration_diagnostics.py
    ├── roman_prune_catalog.py
    └── roman_selection.py
```

---

## Full pipeline walkthrough

### Step 1 — Load and merge  (`step1_load_and_merge.py`)

**Input:** `gaiaXGBoost.fits`, `astra.fits`  
**Output:** `merged.fits`

Loads the Gaia DR3 + XGBoost photometric-parameter catalog and the ASTRA DR1 
spectroscopic catalog and merges them into a single table keyed on Gaia `source_id`. 
This is the widest possible starting point; no cuts are applied here. The output is 
large (~2.4 GB) because it retains all columns from both catalogs.

---

### Step 2 — Roman footprint and stellar cuts  (`step2_footprint_and_cuts.py`)

**Input:** `merged.fits`  
**Output:** `step2_selected.fits`

Two things happen here:

**Footprint cut:** Stars are required to fall inside at least one of the six GBTDS 
overguide tiles. The tile geometry is six rotated rectangles, each 45 × 23 arcmin, 
at a position angle of 90.6°. Centers are:

| Tile | l (deg)   | b (deg)  |
|------|-----------|----------|
| 1    | −0.417948 | −1.200   |
| 2    | −0.008974 | −1.200   |
| 3    | +0.400000 | −1.200   |
| 4    | +0.808974 | −1.200   |
| 5    | +1.217948 | −1.200   |
| 6    |  0.000000 | −0.125   |

**Stellar cuts** (Weiss et al. 2025 §3.3 detectability criteria — defined in the script 
but currently commented out of the keep mask, so only the footprint cut is active):
- T_eff ≤ 5500 K
- log g finite (giant/subgiant selection)
- H ≤ 17 mag (Roman F146 brightness proxy)

The script uses `memmap=True` and reads only the five cut columns before materialising 
any rows, so it handles the 2.4 GB input without loading the full file into memory.

---

### Step 3 — Remote cross-matches  (`step3_remote_xmatch.py`)

**Input:** `step2_selected.fits`  
**Output:** `step3_xmatched.fits`, `step3_summary.txt`

Cross-matches the footprint-selected sample against three external catalogs via the 
CDS XMatch service (astroquery). All cross-matches are left joins keyed on a 
`row_id` assigned at the start of this step, so every input star is retained regardless 
of match success.

| Catalog | VizieR ID | Match radius | Column prefix |
|---------|-----------|-------------|---------------|
| VIRAC2 (Smith et al. 2018) | `II/387/virac2` | 1″ | `vvv_` |
| 2MASS PSC (Skrutskie et al. 2006) | `II/246/out` | 1″ | `tmass_` |
| VVV reddening map (Gonzalez et al. 2012) | `J/A+A/644/A140/ejkmap` | 120″ | `ext_` |

For VIRAC2 and 2MASS, the nearest match within the radius is kept. For the reddening 
map the large radius is intentional: the map cells are ~2 arcmin across, so we are 
matching each star to its enclosing reddening cell, not to a point source. If CDS 
XMatch fails for the reddening table, the script falls back to per-star VizieR 
`query_region` calls.

After matching, this step also runs:
- **Best NIR photometry:** VIRAC2 preferred, 2MASS as fallback, for J, H, Ks
- **Extinction:** E(J−Ks) → A_J, A_H, A_Ks via the Nishiyama bulge law  
  (A_Ks/E(J−Ks) = 0.528, A_H/E(J−Ks) = 0.857, A_J/E(J−Ks) = 1.528)
- **Dereddened magnitudes:** J₀, H₀, Ks₀ and colours
- **Sightline tags:** each star labelled with its VVV reddening-map cell coordinates

---

### Step 4 — Photometry (optional)  (`step4_photometry.py`)

**Input:** `step3_xmatched.fits`  
**Output:** `step4_photometry.fits`

This step re-derives the best NIR photometry, extinction, and dereddened magnitudes 
from the step 3 output using slightly different column-name search logic. It is a 
standalone alternative to running the photometry inside step 3 and exists mainly for 
debugging or if you want to rerun just the photometry without repeating the network 
cross-matches. Step 5 can read either `step3_xmatched.fits` or `step4_photometry.fits`.

---

### Step 5 — Parameter harvesting and final catalog  (`step5_parameters_and_output.py`)

**Input:** `step3_xmatched.fits` (or `step4_photometry.fits`)  
**Output:** `roman_master.fits` / `.csv`, `roman_calibration_ready.fits` / `.csv`, `roman_summary.txt`

Assembles the clean, science-ready catalog by collecting one clearly-labelled column 
per parameter per source, computing luminosity from three independent routes 
(Gaia FLAME, radius × T_eff, and parallax + dereddened G), and writing provenance 
flags (`has_astra`, `has_gspspec`, `has_xgboost`, etc.). Also computes a 
`calibration_ready` boolean for stars with enough constraints to support mixing-length 
calibration (requires T_eff, log g, [M/H], and luminosity).

The `roman_calibration_ready` subset is the direct input to the model calibration 
work in Project C.

---

### TOPCAT step — BDBS cross-match  ⚠️ manual, not scripted

**Input:** `roman_master.fits`, `bdbs.fits`  
**Output:** `roman_master.fits` with `bdbs_` prefixed columns (overwritten in place, or saved separately)

The BDBS optical photometry catalog (Johnson et al. 2022) is cross-matched onto the 
master catalog manually in TOPCAT using a sky match with a ~1″ radius and a left join. 
TOPCAT automatically prefixes the BDBS columns with `bdbs_`. The result is what step 7 
expects as input.

The intermediate file `bdbs_match.csv` in the repo is the BDBS-side match table 
exported from TOPCAT and is kept as a record of which BDBS sources matched.

---

### Step 6 — Diagnostic plots  (`step6_plots.py`)

**Input:** `roman_master.fits`  
**Output:** `plots/01_sky_footprint.png` through `plots/17_reddening_vs_b.png`

Produces 17 publication-quality diagnostic figures covering sky coverage, HR/Kiel 
diagrams, colour-magnitude diagrams, metallicity distributions by sightline, 
parameter completeness, and reddening statistics. See the script docstring for the 
full plot list.

---

### Step 7 — BDBS-derived quantities  (`step7_bdbs.py`)

**Input:** `roman_master.fits` (after TOPCAT BDBS cross-match)  
**Output:** `roman_master_bdbs.fits` / `.csv`, updated plots in `plots/`

Derives quantities from the BDBS optical photometry:
- Dereddened BDBS magnitudes (g₀, r₀, i₀, z₀, y₀, u₀; masks 99.999 sentinel values)
- Dereddened BDBS colours (g−r, g−i, r−i, etc.)
- Optical T_eff from (g−i)₀ via the Casagrande et al. 2010 giant calibration
- Photometric [Fe/H] from RGB locus offset relative to the Zoccali et al. 2003 / 
  Nataf et al. ridge-line grid

Also regenerates plots 03, 10, 11, 12 with BDBS data included, and adds new 
plots 18–21 for BDBS CMD, T_eff comparison, [Fe/H] comparison, and sky map.

---

### Sky map  (`plotssky.py`)

**Input:** `roman_master_bdbs.fits` (falls back to `roman_master.fits`)  
**Output:** `plots/sky_data_coverage.pdf` / `.png`

Produces the publication sky-coverage figure used in the proposal. Each star is 
encoded with four simultaneous visual variables: shape = spectroscopic source 
(ASTRA / GSP-Spec / XGBoost / none), fill = NIR photometry status (dereddened / 
NIR only / none), edge colour = BDBS coverage, size = FLAME age available.

---

## Running the pipeline

```bash
python step1_load_and_merge.py       # ~10–20 min, produces merged.fits
python step2_footprint_and_cuts.py   # fast, produces step2_selected.fits
python step3_remote_xmatch.py        # requires internet; slow for large samples
# (optional) python step4_photometry.py
python step5_parameters_and_output.py

# TOPCAT: left-join bdbs.fits onto roman_master.fits with 1" sky match
#         ensure BDBS columns are prefixed bdbs_
#         save result back as roman_master.fits

python step6_plots.py
python step7_bdbs.py
python plotssky.py
```

---

## Key catalog columns

| Column | Description |
|--------|-------------|
| `sdss_id`, `gaia_dr3_source_id` | Primary identifiers |
| `ra`, `dec`, `l`, `b` | Coordinates (deg) |
| `teff_best`, `teff_source_best` | Best T_eff and its origin |
| `logg_best`, `logg_source_best` | Best log g and its origin |
| `metallicity_best`, `metallicity_source_best` | Best [M/H] and its origin |
| `luminosity_best_lsun`, `log10_lum_best_lsun` | Best luminosity |
| `radius_best_rsun` | Best radius |
| `mass_flame_msun`, `age_flame_gyr` | Gaia FLAME mass and age |
| `j0`, `h0`, `ks0` | Dereddened NIR magnitudes |
| `ext_e_jks` | E(J−Ks) reddening from VVV map |
| `sightline_id`, `sightline_glon`, `sightline_glat` | VVV reddening-map cell |
| `calibration_ready` | Boolean: star has T_eff + log g + [M/H] + luminosity |
| `has_astra`, `has_gspspec`, `has_xgboost`, `has_bdbs`, `has_flame_age` | Provenance flags |

---

## External catalogs used

| Catalog | Reference | Access |
|---------|-----------|--------|
| ASTRA DR1 | Silva et al. 2026 | local file `astra.fits` |
| Gaia DR3 (GSP-Spec, FLAME, distances) | Gaia Collaboration | local file `gaiaXGBoost.fits` |
| XGBoost photometric params | — | embedded in `gaiaXGBoost.fits` |
| VIRAC2 | Smith et al. 2018, VizieR II/387 | CDS XMatch in step 3 |
| 2MASS PSC | Skrutskie et al. 2006, VizieR II/246 | CDS XMatch in step 3 |
| VVV reddening map | Gonzalez et al. 2012, VizieR J/A+A/644/A140 | CDS XMatch in step 3 |
| BDBS | Johnson et al. 2022 | local file `bdbs.fits`, TOPCAT step |
| Bensby+2017 | Bensby et al. 2017 | local file `bensby.fits` |

---

## Dependencies

```
python >= 3.9
numpy
pandas
astropy
astroquery
matplotlib
```

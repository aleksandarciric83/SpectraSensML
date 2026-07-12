# SpectraSensML

**Full-spectrum machine learning for luminescence thermometry and spectral sensing**

SpectraSensML is a desktop application that trains and benchmarks 26+ regression models to predict a continuous variable (temperature, pressure, pH, …) directly from raw luminescence spectra — without manual feature extraction. It is designed for *luminescence thermometry 2.0*: replacing classical Luminescence Intensity Ratio (LIR) methods with data-driven, full-spectrum models.

---

## Features

- **26+ regression models** across five groups: splines/LIR, tree ensembles, kernel methods, regularised linear, and neural networks (MLP + CNN)
- **Automatic preprocessing pipeline**: spectral cropping, background subtraction, SNV / MAX / AREA normalisation
- **PCA dimensionality reduction** with interactive scree and loading plots
- **Sensor fusion**: joint maximum-likelihood inversion in PC space
- **24 publication-ready figures** and matching CSV exports per sub-panel
- **Leaderboard** ranked by average val/test RMSE
- Configurable variable name and unit — works for temperature, pressure, pH, magnetic field, or any scalar measurand
- Supports spectra files with **any delimiter** (tab, space, comma, semicolon — auto-detected)
- Folder naming: `pXXX`/`mXXX` (Celsius offset) or plain integers/decimals (Kelvin directly)

---

## Installation

### Pre-built installers (recommended)

Download the latest release from the [Releases](../../releases) page:

| Platform | File |
|----------|------|
| Windows 10/11 x64 | `SpectraSensML_*_win64_setup.exe` |
| macOS (Intel / Apple Silicon) | `SpectraSensML_*_macos.dmg` |

**macOS note:** the app is not notarized. After dragging to Applications, run once in Terminal:
```bash
xattr -cr "/Applications/SpectraSensML.app"
```

### Linux / from source

Linux users (and anyone who prefers to run from source) can launch the app directly — Python is typically pre-installed:

```bash
git clone https://github.com/aleksandarciric83/SpectraSensML.git
cd SpectraSensML
pip install -r packaging/requirements-lock.txt
python -m lt2_gui
```

---

## Data format

Organise spectra in a root folder with one subfolder per temperature:

```
my_dataset/
  300/    spectrum_001.txt  spectrum_002.txt  ...   # plain Kelvin
  350/    ...
  400/    ...
```

or using the Celsius-offset convention:

```
my_dataset/
  p020/   spectrum_001.txt  ...   # 20 °C → 293.15 K
  m050/   ...                     # −50 °C → 223.15 K
```

Each spectrum file is a two-column text file (wavelength, intensity). Any delimiter works (tab, space, comma, semicolon). Headers are auto-detected; the `Integration Time (sec):` line is used for exposure normalisation if present.

---

## Workflow

1. **Data** — browse to your root folder, load spectra, set folder roles (Train/Val / test_unseen)
2. **Pre-process** — set wavelength window, normalisation, LIR bands
3. **PCA** — run PCA, pick number of components
4. **Models** — select model groups
5. **Run** — choose output directory, start benchmark
6. **Results** — leaderboard + 24 figures + CSV exports

---

## Citation

If you use this software in published work, please cite the accompanying paper. The full reference is shown in the app's **About** tab and stored in `lt2_gui/citation.json`.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE) for details.

**Author:** Aleksandar Ciric — [aleksandar.ciric@ff.bg.ac.rs](mailto:aleksandar.ciric@ff.bg.ac.rs)  
OMAS group · [https://www.omasgroup.org](https://www.omasgroup.org)

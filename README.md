# Lab Repository – FERS-5200 Data Processing

Repository for storing, developing, and maintaining scripts used for the acquisition and analysis of data produced with the **FERS-5200 front-end system** in the context of gamma-camera measurements.

This project contains:
- tools to convert raw FERS-5200 files into ROOT format;
- scripts to analyze ROOT files and extract histograms, pixel maps, spectra, and fitted parameters; 
- utilities to organize data and automatically save results in structured folders. 

---

## Repository Structure


```
Lab_repository/
│
├── Scripts/                         # Main scripts
│   ├── convert_to_root.py           # Converts .txt files to .root format
│   ├── run_pipeline.py              # Runs the full analysis pipeline
│   └── analysis/                    # Analysis modules package
│       ├── event_processing.py      # Event loop, histograms, background subtraction
│       ├── fit_scipy_gmm.py         # Gaussian Mixture Model spectral fit
│       ├── fit_scipy_step.py        # Crystal Ball + step function fit
│       ├── io_manager.py            # Path resolution and JSON/NumPy I/O
│       ├── psf_calc.py              # Point Spread Function (PSF) calculation
│       ├── resolving_power_calc.py  # Multi-source / phantom imaging analysis
│       └── save_results.py          # Saves histograms, maps and plots to disk
│
├── Scripts/tests/                   # pytest test suite
│   ├── test_convert_to_root.py
│   ├── test_event_processing.py
│   ├── test_fit_gmm.py
│   ├── test_fit_scipy_step.py
│   ├── test_io_manager.py
│   ├── test_psf_calc.py
│   ├── test_resolving_power_calc.py
│   └── test_save_results.py
│
├── Data/                            # Save here the .txt files to convert
│   └── Data_converted/              # Placeholder folder for converted .root files
│       └── .gitkeep
│
├── Results/                         # Placeholder for analysis outputs (ignored except .gitkeep)
│   └── .gitkeep
│
├── requirements.txt                 # Python dependencies used by all scripts
└── README.md
```


---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/StefanoSpadano/Lab_repository.git
cd Lab_repository
```

### 2. Create a virtual environment
Create a venv where you just cloned the repository

```bash
python3 -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
All dependencies are listed in requirements.txt file and can be installed via: 

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Convert files from .txt to .root format
FERS-5200 raw data are initially stored in plain .txt format, which is human-readable but inefficient for large datasets, multi-channel acquisition and structured event data.
So we opted to convert them to .root allowing us to store data in a compressed, binary, columnar format optimized for this applications. This ensures:

- much faster I/O and event access;
- structured storage of channels, timestamps and event metadata;
- compatibility with analysis tools such as ROOT, uproot, awkward arrays;
- efficient handling of large datasets over long acquisition periods.

As a result, .root seemed to be the natural choice for analysis workflows in radiation detection and gamma-camera experiments.

During our acquisitions in the laboratory, we thought it was best for us to save the files we gather using the FERS, according to their date of acquisition; so a Run acquired today (01/12/2025) should be saved in a folder called 2025-12-01 and this folder should be placed inside the Data folder.

**Convert the latest date folder automatically** (picks the most recently modified folder in `Data/`):
```bash
python Scripts/convert_to_root.py
```

**Convert a specific date folder:**
```bash
python Scripts/convert_to_root.py --date 2025-12-01
```

**Convert a single file:**
```bash
python Scripts/convert_to_root.py --input-file Data/2025-12-01/Run0_list.txt
```

The `--gain` option (`LG`, `HG`, or `BOTH`) can be added to any of the above calls; if omitted, the gain is inferred from the filename.

The script selects the target folder and creates a corresponding folder inside `Data/Data_converted/` containing the same Runs converted to `.root`.

### 2. Run the analysis pipeline
Once the conversion to .root is complete, the analysis can be performed by running the pipeline script from the repository root, passing the date folder as an argument:
```bash
python Scripts/run_pipeline.py 2025-12-01
```

This pipeline automatically:
- identifies the converted data folder `Data/Data_converted/2025-12-01/`;
- optionally loads a background reference from a `Background/` sub-folder (if present);
- loads each `RunN.root` file;
- performs event processing, histogram extraction and pixel map generation;
- saves all results (plots, JSON histograms, and scatter data) inside a new date-stamped folder under `Results/`.

So, once this has worked correctly an example output could be something like this:
```
Results/
└── 2025-12-01/
    ├── Run0/
    │   ├── histograms/
    │   ├── maps/
    │   └── plots/
    ├── Run1/
    └── ...
```

### 3. Run the spectral fits manually
The fitting steps are kept separate so you can choose the model that best suits each dataset. Run these from the repository root after the pipeline has completed:

```bash
# Crystal Ball + step function fit
python -m analysis.fit_scipy_step 2025-12-01

# Gaussian Mixture Model fit (4 components)
python -m analysis.fit_scipy_gmm 2025-12-01
```

Each script reads the histogram JSON files produced by the pipeline and writes fit plots and a summary JSON (`thesis_summary_data*.json`) into the corresponding `Results/` sub-folders.

### 4. Spatial analysis (PSF and resolving power)
Point Spread Function and multi-source resolving power analyses can be run on individual runs:

```bash
# PSF analysis (single point source)
python -m analysis.psf_calc 2025-12-01 Run0

# Multi-source / phantom imaging analysis
python -m analysis.resolving_power_calc 2025-12-01 Run0
```

---

## Running the tests

The test suite uses [pytest](https://docs.pytest.org/) and covers all analysis modules. Run it from the `Scripts/` directory:

```bash
cd Scripts
python -m pytest tests/ -v
```

To run only a specific test file:
```bash
python -m pytest tests/test_event_processing.py -v
```

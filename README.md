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
├── Scripts/ # Python script to convert files from .txt to .root and Python file to run the pipeline
│ └──analysis # Python scripts for analysis
├── Data/ # save here you data that need to be converted in .root
│ └── Data_converted/ # placeholder folder for converted data 
│ └── .gitkeep
│
├── Results/ # placeholder for analysis outputs (ignored except .gitkeep)
│ └── .gitkeep
│
├── requirements.txt # Python dependencies used by all scripts
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

Once you are in the Scripts folder and you saved your .txt files in the Data folder you can start the conversion by running:
```bash
python convert_to_root.py
```
During our acquisitions in the laboratory, we thought it was best for us to save the files we gather using the FERS, according to their date of acquisition; so a Run acquired today (01/12/2025) should be saved in a folder called 2025-12-01 and this folder should be placed inside the Data folder. 

The script in this way is going to select the latest folder and create a folder, inside the folder Data_converted, with a corresponding name (date) containing the same Runs but converted in .root. 

### 2. Run the analysis pipeline
Once the conversion to .root is complete, the analysis can be performed running the main pipeline script from the Scripts folder:
```bash
python run_pipeline.py
```

This pipeline automatically:
- identifies the latest folder inside Data/Data_converted;
- load each run.root files;
- performs event processing and histogram extraction;
- generate both a global pixel map and a cluster pixel map;
- fits 1D and 2D distributions;
- saves all results (plots, JSON summaries, and derived quantities) inside a new date-stamped folder under Results/ .

So, once this has worked correctly an example output could be something like this:
```
Results/
└── 2025-12-01/
    ├── Run0/
    │   ├── histograms/
    │   ├── fits/
    │   ├── spectrum/
    │   └── plots/
    ├── Run1/
    └── ...
```

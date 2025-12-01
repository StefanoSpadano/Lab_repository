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
│ └── Data_converted/ # Placeholder folder for converted data 
│ └── .gitkeep
│
├── Results/ # Placeholder for analysis outputs (ignored except .gitkeep)
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

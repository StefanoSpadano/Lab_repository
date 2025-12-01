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
├── Data/
│ └── Data_converted/ # Placeholder folder for converted data (kept empty in repo)
│ └── .gitkeep
│
├── Results/ # Placeholder for analysis outputs (ignored except .gitkeep)
│ └── .gitkeep
│
├── requirements.txt # Python dependencies used by all scripts
└── README.md
```




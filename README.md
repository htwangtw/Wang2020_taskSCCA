# TaskSCCA

Author: Hao-Ting Wang

```
├── data                    <- Data
├── models                  <- Trained models
├── notebook                <- analysis
├── references              <- References related to this project etc.
├── reports                 <- Final reports for the study
├── src                     <- Source code for use. Refactor useful things to this folder
│   ├── __init__.py         <- Make src a Python module
│   └── pypma.py            <- SCCA r function wrapper
├── requirements.txt        <- Python package requirements
└── README.md               <- The README for people developing/using this experiment

```

## System requirements

Python:

 - Python >= 3.6
 - pip
 - For other packages, see requirements.txt

R:
 - R >= 3.3.3
 - PMA: 1.0.9
 - impute: 1.48.0
 - plyr: 1.8.4

## install the analysis environment
```
pip install virtualenv

virtualenv env --python=python3.6

# start the virtual environment
source env/bin/activate

# install packages for this project
pip install -r requirements.txt

# exit the virtual enviroment
deactivate
```

When doing analysis or running any scripts in this project:
```
cd /path/to/project/

source env/bin/activate

```
To exit the environment
```
deactivate
```

## Neurosynth decoder
Neurosynth related external data was retrieved on the 20th of December 2018.
The Neurosynth database version is v0.7 (July 2018 release).
The data of 50 topics (v4-topics-50) was retrieved from the neurosynth-web repository on GitHub.

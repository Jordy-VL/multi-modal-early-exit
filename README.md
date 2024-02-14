# An efficient multi-modal multi-exit architecture for document image classification 


![paper figure](https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/images/EE_design.jpg)

## Installation

The scripts require [python >= 3.8](https://www.python.org/downloads/release/python-380/) to run.
We will create a fresh virtualenvironment in which to install all required packages.
```sh
mkvirtualenv -p /usr/bin/python3 EE
```

Using poetry and the readily defined pyproject.toml, we will install all required packages
```sh
workon EE 
pip3 install poetry
poetry install
```

Next step, install those system libraries
```
sudo apt install tesseract-ocr
pip install -q datasets
pip install -q pytesseract
pip install git-lfs
pip install libtesseract-dev
pip install libleptonica-dev
```


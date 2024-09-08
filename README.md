# Multimodal Adaptive Inference for Document Image Classification with Anytime Early Exiting


![paper figure](https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/images/EE_design.jpg)

## Calibration effect on Ramps vs Gates

- **Gates**
  ![gates figure](https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/images/Calibration-effect-gate.drawio.png)

- **Ramps**
  ![ramps figure](https://github.com/Jordy-VL/multi-modal-early-exit/blob/main/EE/images/Calibration-effect-ramps.drawio.png)

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
pip install -q datasets -U
pip install -q pytesseract
sudo apt install git-lfs
sudo apt install libtesseract-dev
sudo apt install libleptonica-dev
```

## How to Contribute

We welcome contributions to the project! Here's how you can get involved:

1. **Fork the Repo**

2. **Create a Branch**: Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3. **Make and Commit Changes**: Commit your changes with a clear and descriptive commit message:
    ```bash
    git commit -m "Add feature: your feature name"
    ```
4. **Push to GitHub**: Push your changes to your forked repository:
    ```bash
    git push origin feature/your-feature-name
    ```
5. **Create a Pull Request**: Go to the original repository and create a pull request from your forked repository. Provide a clear description of your changes and any related issues.

Thank you for your contributions!

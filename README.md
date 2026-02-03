# ⚡ Electric Motor Temperature Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit%20Learn-orange)
![Flask](https://img.shields.io/badge/API-Flask-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

A complete machine learning project for predicting electric motor temperature to enable predictive maintenance in industrial settings.

## 📋 Project Overview

This project predicts electric motor temperatures using operational parameters (current, voltage, RPM, load, etc.) to:
- **Prevent overheating** and equipment failure
- **Optimize maintenance schedules**
- **Improve energy efficiency**
- **Enhance equipment reliability**

## 🏗️ Project Architecture
electric_motor_project/
├── 📁 data/ # Data directory
│ ├── raw/ # Raw datasets
│ └── processed/ # Processed data
├── 📁 notebooks/ # Jupyter notebooks
│ ├── 01_eda.ipynb # Exploratory Data Analysis
│ ├── 02_preprocessing.ipynb # Data preprocessing
│ └── 03_modeling.ipynb # Model training
├── 📁 models/ # Trained ML models
├── 📁 src/ # Source code
├── 📄 app.py # Flask API
├── 📄 dashboard.py # Streamlit dashboard
├── 📄 create_data.py # Data generation
├── 📄 requirements.txt # Dependencies
└── 📄 README.md # This file


## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/electric-motor-temperature-prediction.git
cd electric-motor-temperature-prediction

# Create virtual environment
python -m venv motor_env
motor_env\Scripts\activate  # Windows
# source motor_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
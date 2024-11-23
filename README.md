# Project Delivery 🚀

This guide outlines how to set up and run the project using **bash-only scripts** (no Docker).

## 🛠 Prerequisites
1. Python 3.7+ installed on your system.
2. A Kaggle account to access competition data.
3. `pip` installed for managing Python dependencies.

---

## 🔧 Setup Instructions

### 1. Create and Activate Virtual Environment
Run the following commands to create and activate a virtual environment:
source env/bin/activate

### 2. Install Dependencies
Install the required Python packages by running:
```
pip install -r requirements.txt
```

---

## 🔑 Kaggle Credentials Setup

### 1. Get Kaggle API Key
1. Log in to your Kaggle account.
2. Go to [Account Settings](https://www.kaggle.com/account).
3. Scroll to **API** and click `Create New API Token`.

### 2. Move `kaggle.json` to the Correct Location
1. Download the `kaggle.json` file to your local machine.
2. Move it to the `.kaggle` directory:
```
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🏆 Join the Kaggle Competition
Join the competition by running the following command:
```
kaggle competitions join -c sentiment-analysis-on-movie-reviews
```

Competition link: [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)

---

## 📂 Download Competition Data
After joining the competition, download the dataset by executing:
```
kaggle competitions download -c sentiment-analysis-on-movie-reviews
while [ "`find . -type f -name '*.zip' | wc -l`" -gt 0 ]; do find -type f -name "*.zip" -exec unzip -- '{}' \; -exec rm -- '{}'  \;; done
mkdir dataset
mv *.*v dataset/
```

---

## 📘 Notes
- Ensure the `kaggle.json` file is securely stored and not included in version control.
- Use these steps as a foundation for further enhancements, including Docker integration.



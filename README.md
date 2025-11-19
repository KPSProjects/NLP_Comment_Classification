# Toxic Comment Classification – NLP Coursework

This project is part of the Natural Language Processing module.  
The goal is to build and compare two different NLP pipelines for **toxic comment classification** using the Jigsaw Toxic Comment Classification dataset. The task is a binary classification problem where each comment is labelled as either **toxic (1)** or **non-toxic (0)**.

---

## 1. Project Structure

- `NLP_Toxic_Comments.ipynb` – main Jupyter notebook with all code, explanations and results  
- `train.csv` – training data with comment text and labels  
- `README.md` – this file

---

## 2. Dataset

The dataset used is the **Jigsaw Toxic Comment Classification** dataset from Kaggle:

🔗 **Dataset link:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

The original dataset contains multiple toxicity labels, but in this project only the **binary `toxic`** label was used:

- `0` = non-toxic  
- `1` = toxic  

The notebook includes initial exploratory analysis:
- dataset shape  
- missing values  
- class distribution  
- comment length distribution  

This helps motivate the choice of evaluation metrics such as recall and F1.

---

## 3. Environment & Requirements

The project was developed using:

- Python 3.x  
- Jupyter Notebook  
- Main libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `transformers`
  - `datasets`
  - `torch`
  - `accelerate`

To install the main dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib transformers datasets torch accelerate

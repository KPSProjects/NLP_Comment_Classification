# Toxic Comment Classification – NLP Coursework

This repo is for my **Natural Language Processing** module.  
The aim is to build and compare two NLP pipelines for **toxic comment classification** using the Jigsaw Toxic Comment Classification dataset.

The task is a **binary classification** problem:

- `1` = toxic  
- `0` = non-toxic  

I compare a **classic ML baseline** with a **modern transformer** to see how much difference context-aware models actually make.

---

## 1. What’s in this repo?

- `NLP_Toxic_Comments.ipynb` – main Jupyter notebook with all the code, explanations and plots  
- `train.csv` – training data (Jigsaw toxic comments, needs to be added locally in a `data/` folder if you clone this)  
- `README.md` – this file

---

## 2. Dataset

I use the **Jigsaw Toxic Comment Classification** dataset from Kaggle:

🔗 Dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

The original dataset has multiple toxicity labels (threat, insult, obscene, etc.), but for this coursework I only use the **binary `toxic`** label:

- `0` = non-toxic  
- `1` = toxic  

In the notebook I do some quick exploratory analysis:

- dataset shape (how many rows and columns)  
- check for missing values  
- class distribution (how many toxic vs non-toxic comments)  
- comment length distribution  

This is mainly to show the imbalance in the data and to justify using metrics like **recall** and **F1-score**, not just accuracy.

---

## 3. Models / Pipelines

I compare two different pipelines:

1. **TF–IDF + Logistic Regression**
   - Classic scikit-learn setup  
   - Text is cleaned and converted into TF–IDF vectors  
   - Logistic Regression is trained on these features as a baseline  

2. **DistilBERT (Transformer)**
   - Uses `distilbert-base-uncased` from HuggingFace  
   - Text is tokenised into subword tokens  
   - DistilBERT is fine-tuned on the same toxic vs non-toxic labels using the Trainer API  

Both models are trained and evaluated on the **same train/test split**, and then I compare their performance especially on **toxic recall**.

---

## 4. Environment & Requirements

I worked in **Python 3.x** using **Jupyter Notebook**.

Main libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `transformers`
- `datasets`
- `torch`
- `accelerate`

You can install the core dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib transformers datasets torch accelerate

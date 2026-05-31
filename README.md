# Natural Language Processing with Deep Learning — BIT4133

**Student:** Immaculate Mutheu Muli  
**Reg No:** BSCCS/2024/33678  
**Course:** BSc Computer Science January 2024  
**Lecturer:** Mr. Michael Nyoro  
**Semester:** Fourth Year  


---

## Project — CBK Report Analyser

An NLP system that analyses the **Central Bank of Kenya (CBK) Annual Report 2024/25** using text processing, language modeling, sequence labeling, and deep learning techniques.

**Technologies:** Python · NLTK · spaCy · Gensim (Word2Vec) · pdfplumber · pandas · matplotlib · Google Colab

---

## Weekly Tasks

> ⚠️ GitHub cannot preview `.ipynb` files — click **Open in Colab** to view the full notebook with outputs.

---

### Week 1 — Introduction to NLP and Applications

| Task | File | View |
|------|------|------|
| Mandatory | [Week_1_Mandatory_Tasks.ipynb](./Week_1/Week_1_Mandatory_Tasks.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hrtESW-Y5YYqSp-BnJkgFAWvIPjtw_7i) |
| Extra | [Week_1_Extra_Tasks.ipynb](./Week_1/Week_1_Extra_Tasks.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CFKfco-CEfqZv6alK95NbDt3R2tMBFxG) |

**Topics covered:**
- Python & NLTK setup
- PDF text extraction from CBK Annual Report
- Tokenization (word and sentence)
- Stopwords removal
- Extra: Stemming & lemmatization, word frequency chart, NLU vs NLG

---

### Week 2 — N-gram Models and POS Tagging

| Task | File | View |
|------|------|------|
| Mandatory | [Week_2_Mandatory_Tasks.ipynb](./Week_2/Week_2_Mandatory_Tasks.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zluYV7-0cv5xguOQfPrd3FHqU0XsWcgq) |
| Extra | [Week_2_Extra_Tasks.ipynb](./Week_2/Week_2_Extra_Tasks.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17SQX2Bj45R--UqbjyUPgHUDcSJxTn9Ns) |

**Topics covered:**
- Bigram and trigram models on CBK text
- POS tagging
- Language prediction
- Sentence analysis
- Extra: N-gram frequency chart

---

### Week 3 — Hidden Markov Models and Sequence Labeling

| Task | File | View |
|------|------|------|
| Mandatory | [Week_3_Mandatory_Tasks.ipynb](./Week_3/Week_3_Mandatory_Tasks.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EKnbts8OHG_78LpzBpwBrpkzjf0BSZEY) |
| Extra | [Week_3_Extra_Tasks.ipynb](./Week_3/Week_3_Extra_Tasks.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vCrEvJ-11A85edI1ztSEf-MGXYiYHRCN) |

**Topics covered:**
- Sequence labeling
- HMM transition and emission probabilities
- Named Entity Recognition (NER) — extracting Dr. Kamau Thugge, IMF, Nairobi
- Extra: displaCy NER visualization, HMM trained on CBK data

---

### Week 4 — Syntactic and Semantic Analysis

| Task | File | View |
|------|------|------|
| Mandatory | [Week_4_Mandatory_Tasks.ipynb](./Week_4/Week_4_Mandatory_Tasks.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UCIklrjOs2VZxiTmb-ZPJk5_2m3sEXMk) |

**Topics covered:**
- Dependency parsing using spaCy
- Semantic similarity
- Class assignment: dependency parsing and semantic similarity

---

### Week 5 — Mini NLP Project

| Task | File | View |
|------|------|------|
| Chatbot | [Week_5_Chatbot.ipynb](./Week_5/Week_5_Chatbot.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17M7OKFuXzkV2zYaNhkhedjQd5dCi5Q7O) |

**Project: Student Academic Assistant Chatbot**
- Responds to greetings
- Answers CAT schedule questions
- Answers unit registration questions
- Exits correctly

---

## Repo Structure

```
Natural_Language_Processing/
├── Week_1/
│   ├── Week_1_Mandatory_Tasks.ipynb
│   └── Week_1_Extra_Tasks.ipynb
├── Week_2/
│   ├── Week_2_Mandatory_Tasks.ipynb
│   └── Week_2_Extra_Tasks.ipynb
├── Week_3/
│   ├── Week_3_Mandatory_Tasks.ipynb
│   └── Week_3_Extra_Tasks.ipynb
├── Week_4/
│   └── Week_4_Mandatory_Tasks.ipynb
├── Week_5/
│   └── Week_5_Chatbot.ipynb
└── README.md
```

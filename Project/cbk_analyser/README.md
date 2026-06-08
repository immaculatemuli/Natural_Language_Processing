# CBK Report Analyser — NLP Web Application

A web application that analyses Central Bank of Kenya (CBK) PDF reports using NLP techniques including Tokenization, Stopword Removal, Lemmatization, N-gram Models, POS Tagging, Named Entity Recognition, and Semantic Similarity.

## Setup & Run

**Step 1: Install Python dependencies**
```
pip install -r requirements.txt
```

**Step 2: Download spaCy English model**
```
python -m spacy download en_core_web_sm
```

**Step 3: Start the Flask server**
```
python app.py
```

**Step 4: Open in browser**
```
http://localhost:5000
```

## Features

1. **PDF Upload** — Upload any CBK PDF report; text is extracted automatically
2. **Text Overview** — Tokenization, stopword removal, and lemmatization stats
3. **Key Figures** — Central Bank Rate, Inflation, GDP Growth, FX Reserves
4. **N-gram Analysis** — Top bigrams and trigrams with bar chart
5. **POS Tagging** — Part-of-speech tags for the first 30 meaningful tokens
6. **Named Entities** — PERSON, ORG, GPE, DATE, PERCENT, MONEY, CARDINAL
7. **Word Frequency** — Top 20 words visualised as a bar chart
8. **Semantic Search** — Find the most relevant sentences for any query

## NLP Concepts Demonstrated

| Concept | Library | Where Used |
|---|---|---|
| Tokenization | NLTK | Text Overview |
| Stopword Removal | NLTK | Text Overview, Word Frequency |
| Lemmatization | NLTK WordNetLemmatizer | Text Overview |
| N-gram Models | NLTK bigrams/trigrams | N-gram Analysis |
| POS Tagging | NLTK pos_tag | POS Tags section |
| Named Entity Recognition | spaCy | Named Entities section |
| Semantic Similarity | spaCy word vectors | Semantic Search |

## Project Structure

```
cbk_analyser/
├── app.py          — Flask routes and API endpoints
├── analyser.py     — All NLP processing logic
├── templates/
│   └── index.html  — Single-page dashboard (HTML + JS)
├── static/
│   └── style.css   — Dashboard styling
├── requirements.txt
└── README.md
```

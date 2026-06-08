"""
analyser.py — NLP processing for CBK Report Analyser
"""

import re
import nltk
import spacy
import PyPDF2
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import bigrams, trigrams
from nltk import pos_tag

def _download_nltk():
    for path, name in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

_download_nltk()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

STOP_WORDS  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

_NER_DISABLED = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
_VEC_DISABLED = ["tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]

POS_MEANINGS = {
    "CC":"Coordinating conjunction","CD":"Cardinal number","DT":"Determiner",
    "IN":"Preposition","JJ":"Adjective","JJR":"Adjective, comparative",
    "JJS":"Adjective, superlative","MD":"Modal verb","NN":"Noun",
    "NNS":"Noun, plural","NNP":"Proper noun","NNPS":"Proper noun, plural",
    "PRP":"Personal pronoun","PRP$":"Possessive pronoun","RB":"Adverb",
    "RBR":"Adverb, comparative","TO":"to","UH":"Interjection",
    "VB":"Verb","VBD":"Verb, past tense","VBG":"Verb, gerund",
    "VBN":"Verb, past participle","VBP":"Verb, present","VBZ":"Verb, 3rd person",
    "WDT":"Wh-determiner","WP":"Wh-pronoun","WRB":"Wh-adverb",
}

NER_LABELS = ["PERSON","ORG","GPE","DATE","PERCENT","MONEY","CARDINAL"]

# Questions auto-run against the report to generate plain-language highlights
AUTO_INSIGHTS = {
    "interest_rate": "What is the central bank rate monetary policy interest rate decision?",
    "inflation":     "What is the inflation rate consumer prices cost of living?",
    "gdp_growth":    "What is the economic growth GDP forecast projection?",
    "forex":         "What are the foreign exchange reserves currency position?",
    "banking":       "What are the banking sector developments financial stability?",
    "government":    "What government fiscal policy budget measures were announced?",
    "credit":        "What is happening with private sector credit lending loans?",
    "employment":    "What is happening with employment jobs labour market?",
}


# ── PDF ──────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_stream):
    reader = PyPDF2.PdfReader(file_stream)
    page_count = len(reader.pages)
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts), page_count


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_text(text):
    original_word_count = len(text.split())
    all_alpha    = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    filtered_all = [w for w in all_alpha if w.lower() not in STOP_WORDS]

    sample_tokens   = word_tokenize(text[:3000])
    sample_alpha    = [t for t in sample_tokens if t.isalpha()]
    sample_filtered = [t for t in sample_alpha if t.lower() not in STOP_WORDS]
    sample_lemma    = [lemmatizer.lemmatize(t.lower()) for t in sample_filtered]

    return {
        "original_word_count": original_word_count,
        "token_count":         len(all_alpha),
        "filtered_count":      len(filtered_all),
        "stopwords_removed":   len(all_alpha) - len(filtered_all),
        "before_examples":     sample_alpha[:10],
        "after_examples":      sample_filtered[:10],
        "lemma_examples":      [
            {"original": sample_filtered[i], "lemma": sample_lemma[i]}
            for i in range(min(10, len(sample_filtered)))
        ],
        "filtered_tokens":     filtered_all,
    }


# ── N-grams ───────────────────────────────────────────────────────────────────
def get_ngrams(tokens):
    lower  = [t.lower() for t in tokens]
    bgs    = Counter(bigrams(lower)).most_common(10)
    tgs    = Counter(trigrams(lower)).most_common(5)
    return {
        "bigrams":  [{"phrase":" ".join(b),"count":c} for b,c in bgs],
        "trigrams": [{"phrase":" ".join(t),"count":c} for t,c in tgs],
    }


# ── POS ───────────────────────────────────────────────────────────────────────
def get_pos_tags(tokens):
    return [
        {"word":w,"tag":t,"meaning":POS_MEANINGS.get(t,"Other")}
        for w,t in pos_tag(tokens[:30])
    ]


# ── NER ───────────────────────────────────────────────────────────────────────
def get_named_entities(text):
    sample  = text[:12000]
    chunks  = [sample[i:i+6000] for i in range(0, len(sample), 6000)]
    by_label = {l:[] for l in NER_LABELS}
    for doc in nlp.pipe(chunks, disable=_NER_DISABLED):
        for ent in doc.ents:
            if ent.label_ in by_label:
                by_label[ent.label_].append(ent.text.strip())
    return {
        lbl: {"count": len(ents), "examples": list(dict.fromkeys(ents))[:5]}
        for lbl, ents in by_label.items()
    }


# ── Key figures ───────────────────────────────────────────────────────────────
def _find(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"

def _find_raw(pattern, text):
    """Return the full match group(1) as a float string, or None."""
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).replace(",", "").strip()
    try:
        return str(float(val))
    except ValueError:
        return None

def extract_key_figures(text):
    # Central Bank Rate
    cbr = _find(r"central\s+bank\s+rate[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if cbr == "N/A":
        cbr = _find(r"(?:CBR|policy\s+rate)[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if cbr == "N/A":
        cbr = _find(r"maintained[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)

    # Inflation — prefer headline/overall, avoid "below X%"-style forward guidance
    inf = _find(r"(?:overall|headline)\s+inflation[^a-z]{0,5}(?:was|stood|remained|eased|rose|increased|decreased)[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if inf == "N/A":
        inf = _find(r"(?:overall|headline)\s+inflation[^0-9]{0,40}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if inf == "N/A":
        inf = _find(r"inflation[^a-z]{0,5}(?:was|stood|remained|eased|rose)[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if inf == "N/A":
        inf = _find(r"inflation[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)

    # GDP growth
    gdp = _find(r"(?:GDP|economy|economic\s+growth)[^a-z]{0,10}(?:grew|projected|estimated|forecast)[^0-9]{0,40}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if gdp == "N/A":
        gdp = _find(r"(?:GDP|economic)\s+growth[^0-9]{0,40}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if gdp == "N/A":
        gdp = _find(r"growth[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)

    # Forex reserves — look for months of import cover first (more reliably extracted)
    cover = _find(r"([\d]+\.?[\d]*)\s*months?\s*(?:of)?\s*(?:import|import\s+cover)", text)
    res   = _find(r"(?:foreign\s+exchange|forex|fx)\s+reserves[^0-9]{0,60}([\d,]+\.?[\d]*)\s*(?:billion)?", text)
    if res == "N/A":
        res = _find(r"reserves[^0-9]{0,40}([\d,]+\.?[\d]*)\s*billion", text)

    # Raw numeric values (for charts)
    cbr_num  = None if cbr  == "N/A" else cbr
    inf_num  = None if inf  == "N/A" else inf
    gdp_num  = None if gdp  == "N/A" else gdp
    res_num  = None if res  == "N/A" else res.replace(",","")

    return {
        "cbr":         f"{cbr}%"   if cbr != "N/A" else "N/A",
        "inflation":   f"{inf}%"   if inf != "N/A" else "N/A",
        "gdp_growth":  f"{gdp}%"   if gdp != "N/A" else "N/A",
        "fx_reserves": f"USD {res}B" if res != "N/A" else "N/A",
        "import_cover": f"{cover} months" if cover != "N/A" else "N/A",
        # Raw numbers for chart rendering on frontend
        "cbr_num":    float(cbr_num)  if cbr_num  else None,
        "inf_num":    float(inf_num)  if inf_num  else None,
        "gdp_num":    float(gdp_num)  if gdp_num  else None,
        "res_num":    float(res_num)  if res_num  else None,
    }


# ── Report period extraction ───────────────────────────────────────────────────
def extract_report_period(text):
    """Extract the reporting period (month+year or quarter+year) from the report text."""
    sample = text[:4000]
    # Month + Year (most specific)
    m = re.search(
        r'(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+(20\d{2})',
        sample, re.IGNORECASE
    )
    if m:
        return m.group(0).title()
    # Quarter
    m = re.search(r'(Q[1-4])\s+(20\d{2})', sample, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    # Year range
    m = re.search(r'(20\d{2})[/\-](20\d{2})', sample)
    if m:
        return m.group(0)
    # Plain year
    m = re.search(r'\b(20\d{2})\b', sample)
    if m:
        return m.group(1)
    return None


# ── Word frequency ────────────────────────────────────────────────────────────
def get_word_frequency(tokens, top_n=20):
    counts = Counter(t.lower() for t in tokens).most_common(top_n)
    return [{"word":w,"count":c} for w,c in counts]


# ── Sentence vectorisation (shared helper) ────────────────────────────────────
def _vectorise_sentences(text, max_sents=80, min_words=6, char_cap=10000):
    """Tokenise, filter, and batch-vectorise sentences. Returns (sentences, docs)."""
    sents = [
        s.strip() for s in sent_tokenize(text[:char_cap])
        if len(s.split()) >= min_words and any(c.isalpha() for c in s)
    ][:max_sents]
    docs = list(nlp.pipe([s[:400] for s in sents], disable=_VEC_DISABLED, batch_size=32))
    return sents, docs


def _score(qdoc, question, sents, sent_docs):
    results = []
    for sent, sdoc in zip(sents, sent_docs):
        if qdoc.has_vector and sdoc.has_vector:
            sim = float(qdoc.similarity(sdoc))
        else:
            qw  = set(question.lower().split())
            sw  = set(sent.lower().split())
            sim = len(qw & sw) / max(len(qw), 1)
        results.append((sent, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ── Semantic search (single query) — reuses cached vectors when available ─────
def semantic_search(query, text, top_n=3, cached=None):
    if cached:
        sents, sent_docs = cached
    else:
        sents, sent_docs = _vectorise_sentences(text, max_sents=80)
    qdoc    = nlp(query)
    results = _score(qdoc, query, sents, sent_docs)
    return [{"sentence": s, "score": round(sc, 4)} for s, sc in results[:top_n]]


# ── Auto-insights: one vectorisation pass for all 8 queries ───────────────────
def build_auto_insights(text):
    sents, sent_docs = _vectorise_sentences(text, max_sents=80, char_cap=10000)
    insights = {}
    # Pre-create all query docs in a single pipe call
    questions = list(AUTO_INSIGHTS.values())
    q_docs    = list(nlp.pipe(questions, disable=_VEC_DISABLED, batch_size=8))

    for (key, question), qdoc in zip(AUTO_INSIGHTS.items(), q_docs):
        results = _score(qdoc, question, sents, sent_docs)
        if results and results[0][1] >= 0.25:
            best_sent, best_score = results[0]
            insights[key] = {"sentence": best_sent, "score": round(best_score, 4)}

    return insights, (sents, sent_docs)   # return cache for reuse by search


# ── Master ────────────────────────────────────────────────────────────────────
def analyse_pdf(file_stream):
    raw_text, page_count = extract_text_from_pdf(file_stream)
    pp     = preprocess_text(raw_text)
    tokens = pp["filtered_tokens"]

    auto_insights, sent_cache = build_auto_insights(raw_text)

    return {
        "page_count":     page_count,
        "period":         extract_report_period(raw_text),
        "preprocessing":  {k: pp[k] for k in [
            "original_word_count","token_count","filtered_count",
            "stopwords_removed","before_examples","after_examples","lemma_examples"
        ]},
        "key_figures":    extract_key_figures(raw_text),
        "auto_insights":  auto_insights,
        "ngrams":         get_ngrams(tokens),
        "pos_tags":       get_pos_tags(tokens),
        "named_entities": get_named_entities(raw_text),
        "word_frequency": get_word_frequency(tokens),
        "_sent_cache":    sent_cache,   # kept in memory for fast search
    }

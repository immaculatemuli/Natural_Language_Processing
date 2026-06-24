"""
analyser.py — NLP engine for CBK Mwananchi Economic Assistant
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
        ("tokenizers/punkt",                     "punkt"),
        ("tokenizers/punkt_tab",                 "punkt_tab"),
        ("corpora/stopwords",                    "stopwords"),
        ("corpora/wordnet",                      "wordnet"),
        ("taggers/averaged_perceptron_tagger",    "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng","averaged_perceptron_tagger_eng"),
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
_VEC_DISABLED = ["tagger", "parser", "ner",   "attribute_ruler", "lemmatizer"]

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


# ── PDF ───────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_stream):
    reader    = PyPDF2.PdfReader(file_stream)
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
        "filtered_tokens": filtered_all,
    }


# ── N-grams ───────────────────────────────────────────────────────────────────
def get_ngrams(tokens):
    lower = [t.lower() for t in tokens]
    bgs   = Counter(bigrams(lower)).most_common(10)
    tgs   = Counter(trigrams(lower)).most_common(5)
    return {
        "bigrams":  [{"phrase": " ".join(b), "count": c} for b, c in bgs],
        "trigrams": [{"phrase": " ".join(t), "count": c} for t, c in tgs],
    }


# ── POS ───────────────────────────────────────────────────────────────────────
def get_pos_tags(tokens):
    return [
        {"word": w, "tag": t, "meaning": POS_MEANINGS.get(t, "Other")}
        for w, t in pos_tag(tokens[:30])
    ]


# ── NER ───────────────────────────────────────────────────────────────────────
def get_named_entities(text):
    sample   = text[:12000]
    chunks   = [sample[i:i+6000] for i in range(0, len(sample), 6000)]
    by_label = {l: [] for l in NER_LABELS}
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

def extract_key_figures(text):
    cbr = _find(r"central\s+bank\s+rate[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if cbr == "N/A":
        cbr = _find(r"(?:CBR|policy\s+rate)[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if cbr == "N/A":
        cbr = _find(r"maintained[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)

    inf = _find(r"(?:overall|headline)\s+inflation[^a-z]{0,5}(?:was|stood|remained|eased|rose|increased|decreased)[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if inf == "N/A":
        inf = _find(r"(?:overall|headline)\s+inflation[^0-9]{0,40}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if inf == "N/A":
        inf = _find(r"inflation[^a-z]{0,5}(?:was|stood|remained|eased|rose)[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if inf == "N/A":
        inf = _find(r"inflation[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)

    gdp = _find(r"(?:GDP|economy|economic\s+growth)[^a-z]{0,10}(?:grew|projected|estimated|forecast)[^0-9]{0,40}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if gdp == "N/A":
        gdp = _find(r"(?:GDP|economic)\s+growth[^0-9]{0,40}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)
    if gdp == "N/A":
        gdp = _find(r"growth[^0-9]{0,30}([\d]+\.?[\d]*)\s*(?:per\s*cent|%)", text)

    cover = _find(r"([\d]+\.?[\d]*)\s*months?\s*(?:of)?\s*(?:import|import\s+cover)", text)
    res   = _find(r"(?:foreign\s+exchange|forex|fx)\s+reserves[^0-9]{0,60}([\d,]+\.?[\d]*)\s*(?:billion)?", text)
    if res == "N/A":
        res = _find(r"reserves[^0-9]{0,40}([\d,]+\.?[\d]*)\s*billion", text)

    cbr_num = None if cbr == "N/A" else cbr
    inf_num = None if inf == "N/A" else inf
    gdp_num = None if gdp == "N/A" else gdp
    res_num = None if res == "N/A" else res.replace(",", "")

    return {
        "cbr":          f"{cbr}%"      if cbr != "N/A" else "N/A",
        "inflation":    f"{inf}%"      if inf != "N/A" else "N/A",
        "gdp_growth":   f"{gdp}%"      if gdp != "N/A" else "N/A",
        "fx_reserves":  f"USD {res}B"  if res != "N/A" else "N/A",
        "import_cover": f"{cover} months" if cover != "N/A" else "N/A",
        "cbr_num":      float(cbr_num) if cbr_num else None,
        "inf_num":      float(inf_num) if inf_num else None,
        "gdp_num":      float(gdp_num) if gdp_num else None,
        "res_num":      float(res_num) if res_num else None,
    }


# ── Report period ─────────────────────────────────────────────────────────────
def extract_report_period(text):
    sample = text[:4000]
    m = re.search(
        r'(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\s+(20\d{2})',
        sample, re.IGNORECASE
    )
    if m: return m.group(0).title()
    m = re.search(r'(Q[1-4])\s+(20\d{2})', sample, re.IGNORECASE)
    if m: return m.group(0).upper()
    m = re.search(r'(20\d{2})[/\-](20\d{2})', sample)
    if m: return m.group(0)
    m = re.search(r'\b(20\d{2})\b', sample)
    if m: return m.group(1)
    return None


# ── Word frequency ────────────────────────────────────────────────────────────
def get_word_frequency(tokens, top_n=20):
    counts = Counter(t.lower() for t in tokens).most_common(top_n)
    return [{"word": w, "count": c} for w, c in counts]


# ── Sentence vectorisation ────────────────────────────────────────────────────
def _vectorise_sentences(text, max_sents=80, min_words=6, char_cap=10000):
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


def semantic_search(query, text, top_n=3, cached=None):
    sents, sent_docs = cached if cached else _vectorise_sentences(text, max_sents=80)
    qdoc    = nlp(query)
    results = _score(qdoc, query, sents, sent_docs)
    return [{"sentence": s, "score": round(sc, 4)} for s, sc in results[:top_n]]


def build_auto_insights(text):
    sents, sent_docs = _vectorise_sentences(text, max_sents=80, char_cap=10000)
    insights  = {}
    questions = list(AUTO_INSIGHTS.values())
    q_docs    = list(nlp.pipe(questions, disable=_VEC_DISABLED, batch_size=8))

    for (key, question), qdoc in zip(AUTO_INSIGHTS.items(), q_docs):
        results = _score(qdoc, question, sents, sent_docs)
        if results and results[0][1] >= 0.25:
            best_sent, best_score = results[0]
            insights[key] = {"sentence": best_sent, "score": round(best_score, 4)}

    return insights, (sents, sent_docs)


# ── Master analysis ───────────────────────────────────────────────────────────
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
        "_sent_cache":    sent_cache,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NEW FEATURES — AI Assistant, Health Score, Cost of Living, Predictive Text
# ═══════════════════════════════════════════════════════════════════════════════

QUESTION_SUGGESTIONS = [
    "Why is inflation going up in Kenya?",
    "Why is the shilling weakening against the dollar?",
    "Should I take a bank loan now?",
    "What is the Central Bank Rate and what does it mean?",
    "How does inflation affect my salary?",
    "Is it a good time to save money?",
    "What is the economic growth rate?",
    "What is the government doing about debt?",
    "Should I invest in Treasury Bills now?",
    "Why is the cost of living so high?",
    "How can I protect my savings from inflation?",
    "What is the forex reserve situation?",
    "Why are fuel prices high?",
    "What does monetary policy mean for ordinary Kenyans?",
    "How is the banking sector doing?",
    "How does the CBR affect my loan payments?",
    "Is the Kenyan economy growing?",
    "What does import cover mean?",
    "How do interest rate changes affect businesses?",
    "What is GDP and why does it matter to me?",
    "Kwa nini bei zinaongezeka?",
    "Je, ni wakati mzuri wa kukopa?",
    "Shilingi inabadilika vipi dhidi ya dola?",
    "Nini kinasababisha mfumuko wa bei?",
    "Je, akiba yangu iko salama?",
    "Kiwango cha riba kinaathiri vipi mkopo wangu?",
]


def detect_language(text):
    """Heuristic: if 2+ Swahili words found, treat as Swahili."""
    sw_words = {
        "kwa","nini","je","na","ya","za","wa","ni","au","lakini",
        "shilingi","bei","pesa","benki","uchumi","mkopo","akiba",
        "mfumuko","riba","kiwango","kazi","bidhaa","serikali",
    }
    words = set(text.lower().split())
    return "sw" if len(words & sw_words) >= 2 else "en"


def get_predictive_suggestions(partial_text):
    """Return up to 5 question completions for the given partial input."""
    if len(partial_text) < 2:
        return []
    lower = partial_text.lower()
    return [q for q in QUESTION_SUGGESTIONS if lower in q.lower()][:5]


def calculate_health_score(key_figures):
    """Return economic health score 0-100 with traffic-light breakdown."""
    kf     = key_figures or {}
    scores = []
    indicators = {}

    cbr = kf.get("cbr_num")
    inf = kf.get("inf_num")
    gdp = kf.get("gdp_num")
    res = kf.get("res_num")

    if cbr is not None:
        if   cbr <= 9:  s, lbl, col = 90, "Healthy",  "green"
        elif cbr <= 11: s, lbl, col = 60, "Moderate", "yellow"
        else:           s, lbl, col = 30, "High",     "red"
        scores.append(s)
        indicators["cbr"] = {"value": f"{cbr}%", "score": s, "label": lbl, "color": col, "name": "Central Bank Rate"}

    if inf is not None:
        if   2.5 <= inf <= 7.5: s, lbl, col = 90, "On target",   "green"
        elif inf <= 10:         s, lbl, col = 55, "Above target", "yellow"
        else:                   s, lbl, col = 25, "Very high",    "red"
        scores.append(s)
        indicators["inflation"] = {"value": f"{inf}%", "score": s, "label": lbl, "color": col, "name": "Inflation Rate"}

    if gdp is not None:
        if   gdp >= 5: s, lbl, col = 90, "Strong growth",   "green"
        elif gdp >= 3: s, lbl, col = 65, "Moderate growth", "yellow"
        else:          s, lbl, col = 30, "Slow growth",     "red"
        scores.append(s)
        indicators["gdp"] = {"value": f"{gdp}%", "score": s, "label": lbl, "color": col, "name": "GDP Growth"}

    if res is not None:
        if   res >= 7: s, lbl, col = 90, "Strong",   "green"
        elif res >= 5: s, lbl, col = 65, "Adequate", "yellow"
        else:          s, lbl, col = 35, "Low",       "red"
        scores.append(s)
        indicators["forex"] = {"value": f"USD {res}B", "score": s, "label": lbl, "color": col, "name": "Forex Reserves"}

    overall = round(sum(scores) / len(scores)) if scores else 0
    if   overall >= 70: overall_lbl, overall_col = "Economy is stable",      "green"
    elif overall >= 45: overall_lbl, overall_col = "Some warning signs",     "yellow"
    else:               overall_lbl, overall_col = "Economy under pressure",  "red"

    return {
        "score":      overall,
        "label":      overall_lbl,
        "color":      overall_col,
        "indicators": indicators,
    }


def calculate_cost_of_living_impact(salary, key_figures):
    """Show how the current inflation rate eats into a given monthly salary."""
    kf  = key_figures or {}
    inf = kf.get("inf_num")
    if not inf or salary <= 0:
        return None

    # Approximate Kenyan household spending distribution
    budget = [
        {"category": "Food & Groceries",  "pct": 38, "impact": 1.25},
        {"category": "Rent & Utilities",  "pct": 28, "impact": 0.80},
        {"category": "Transport & Fuel",  "pct": 14, "impact": 1.35},
        {"category": "Healthcare",        "pct":  8, "impact": 1.00},
        {"category": "Education & Other", "pct": 12, "impact": 0.90},
    ]

    items, total_increase = [], 0.0
    for b in budget:
        amount         = salary * b["pct"] / 100
        effective_inf  = inf * b["impact"]
        increase       = amount * effective_inf / 100
        items.append({
            "category": b["category"],
            "before":   round(amount),
            "after":    round(amount + increase),
            "increase": round(increase),
            "pct":      b["pct"],
        })
        total_increase += increase

    return {
        "salary":               salary,
        "inflation":            inf,
        "total_extra_cost":     round(total_increase),
        "real_value":           round(salary - total_increase),
        "purchasing_power_loss": round((total_increase / salary) * 100, 1),
        "items":                items,
    }


def generate_chat_response(message, search_results, key_figures, lang="en"):
    """Generate an intelligent, context-aware chatbot response."""
    ml = message.lower()
    kf = key_figures or {}
    cbr = kf.get("cbr_num")
    inf = kf.get("inf_num")
    gdp = kf.get("gdp_num")
    fx  = kf.get("fx_reserves", "N/A")
    cvr = kf.get("import_cover", "N/A")

    INTENTS = {
        "loan":      ["loan","borrow","interest","credit","lending","debt","mkopo","kukopa","riba","kopi"],
        "inflation": ["inflation","prices","expensive","cost of living","food","fuel","bei","ghali","mfumuko","gharama"],
        "forex":     ["shilling","dollar","exchange","forex","currency","usd","ksh","shilingi","dola","fedha za"],
        "gdp":       ["economy","growth","gdp","jobs","employment","business","uchumi","ukuaji","kazi","ajira"],
        "savings":   ["save","savings","invest","treasury","t-bill","bond","akiba","kuweka","uwekezaji"],
    }

    intent = "general"
    for key, words in INTENTS.items():
        if any(w in ml for w in words):
            intent = key
            break

    def r(english, swahili):
        return swahili if lang == "sw" else english

    resp = ""

    if intent == "loan" and cbr is not None:
        lo, hi = f"{cbr+3:.1f}", f"{cbr+5:.1f}"
        if cbr >= 11:
            resp = r(
                f"The CBK rate is {kf.get('cbr')}, meaning bank loans cost around {lo}%–{hi}% per year — quite high by Kenyan standards. "
                f"If the loan isn't urgent, consider waiting for rates to ease. Compare multiple banks, and look at SACCOs — they often lend at lower rates than commercial banks.",
                f"Kiwango cha CBK ni {kf.get('cbr')}, yaani mikopo ya benki inagharimu takriban {lo}%–{hi}% kwa mwaka — ghali sana kwa viwango vya Kenya. "
                f"Ikiwa mkopo si wa haraka, subiri viwango vipungue. Linganisha benki nyingi na angalia SACCO — mara nyingi zinakopesha kwa viwango vya chini."
            )
        elif cbr >= 9:
            resp = r(
                f"The CBK rate is {kf.get('cbr')}, placing bank loans at roughly {lo}%–{hi}% per year. Rates are moderate. "
                f"Shop around between banks — rates can vary significantly. A SACCO might give you a better deal.",
                f"Kiwango cha CBK ni {kf.get('cbr')}, hivyo mikopo ya benki ni takriban {lo}%–{hi}% kwa mwaka. Viwango ni wastani. "
                f"Linganisha benki mbalimbali — viwango vinatofautiana sana. SACCO inaweza kukupa masharti bora zaidi."
            )
        else:
            resp = r(
                f"The CBK rate is {kf.get('cbr')} — relatively favourable. Bank loans run around {lo}%–{hi}% per year. "
                f"This is a reasonable time to borrow for productive investments like housing, education, or business expansion.",
                f"Kiwango cha CBK ni {kf.get('cbr')} — ni nzuri kwa viwango vya Kenya. Mikopo ya benki ni takriban {lo}%–{hi}% kwa mwaka. "
                f"Hii ni wakati mzuri wa kukopa kwa uwekezaji wa uzalishaji kama nyumba, elimu, au upanuzi wa biashara."
            )

    elif intent == "inflation" and inf is not None:
        groceries_now = int(5000 * (1 + inf / 100))
        if inf > 7.5:
            resp = r(
                f"Inflation is at {kf.get('inflation')} — above CBK's 7.5% ceiling. Prices are rising faster than the government's comfort zone. "
                f"A grocery basket that cost Ksh 5,000 now costs about Ksh {groceries_now:,}. "
                f"To protect yourself: cut non-essential spending, move savings to a Money Market Fund, and buy staple foods in bulk when possible.",
                f"Mfumuko wa bei uko {kf.get('inflation')} — juu ya kiwango cha juu cha CBK cha 7.5%. Bei zinaongezeka haraka kuliko inavyotarajiwa. "
                f"Bidhaa za mboga zilizokuwa na bei ya Ksh 5,000 sasa zinagharimu takriban Ksh {groceries_now:,}. "
                f"Ili kujilinda: punguza matumizi yasiyohitajika, hifadhi akiba kwenye Mfuko wa Soko la Pesa, na nunua vyakula vya msingi kwa wingi."
            )
        elif inf >= 2.5:
            resp = r(
                f"Inflation is at {kf.get('inflation')} — within CBK's 2.5%–7.5% target. Prices are rising, but at a manageable pace. "
                f"Your Ksh 5,000 grocery budget now costs about Ksh {groceries_now:,}. Make sure any salary negotiation accounts for this.",
                f"Mfumuko wa bei uko {kf.get('inflation')} — ndani ya kikomo cha CBK cha 2.5%–7.5%. Bei zinaongezeka lakini kwa kasi inayodhibitiwa. "
                f"Bajeti yako ya Ksh 5,000 ya mboga sasa inagharimu takriban Ksh {groceries_now:,}. Hakikisha mazungumzo yoyote ya mshahara yanazingatia hili."
            )
        else:
            resp = r(
                f"Inflation is very low at {kf.get('inflation')} — below CBK's 2.5% floor. Prices are very stable, which is great for purchasing power. "
                f"However, very low inflation can indicate weak economic demand, so CBK may consider stimulus measures.",
                f"Mfumuko wa bei ni mdogo sana kwa {kf.get('inflation')} — chini ya kikomo cha chini cha CBK cha 2.5%. Bei ni imara sana, ambayo ni nzuri. "
                f"Hata hivyo, mfumuko wa bei mdogo sana unaweza kuonyesha mahitaji dhaifu ya uchumi, hivyo CBK inaweza kuchukua hatua za kuchochea."
            )

    elif intent == "forex":
        cover_str = f" ({cvr} of import cover)" if cvr != "N/A" else ""
        resp = r(
            f"Kenya holds {fx} in foreign exchange reserves{cover_str}. CBK's minimum target is 4 months of import cover. "
            f"These reserves keep the shilling stable and allow Kenya to pay for critical imports — fuel, medicines, industrial equipment. "
            f"Strong reserves generally mean a more predictable exchange rate, which keeps imported goods prices stable.",
            f"Kenya ina {fx} katika akiba ya fedha za kigeni{cover_str}. Lengo la chini la CBK ni miezi 4 ya uingizaji. "
            f"Akiba hizi zinadumisha thamani ya shilingi na kuruhusu Kenya kulipa bidhaa muhimu zilizoagizwa — mafuta, dawa, vifaa vya viwanda. "
            f"Akiba kubwa kwa ujumla zinamaanisha kiwango cha ubadilishaji kinachoweza kutabirika zaidi, ambacho kinadumisha bei imara za bidhaa zilizoagizwa."
        )

    elif intent == "gdp" and gdp is not None:
        if gdp >= 5:
            resp = r(
                f"GDP growth is strong at {kf.get('gdp_growth')} — at or above Kenya's potential. Businesses are expanding, government is collecting more tax revenue "
                f"for public services, and job opportunities are growing. A good climate for career advancement and business investment.",
                f"Ukuaji wa GDP ni imara kwa {kf.get('gdp_growth')} — sawa na au juu ya uwezo wa Kenya. Biashara zinaenea, serikali inakusanya kodi zaidi "
                f"kwa ajili ya huduma za umma, na fursa za kazi zinaongezeka. Hali nzuri kwa ukuaji wa kazi na uwekezaji wa biashara."
            )
        elif gdp >= 3:
            resp = r(
                f"GDP growth is {kf.get('gdp_growth')} — positive, but below Kenya's 5–6% potential. The economy is moving, "
                f"but not fast enough to absorb the 800,000+ new job seekers entering the market each year. Job competition may be tighter than normal.",
                f"Ukuaji wa GDP ni {kf.get('gdp_growth')} — chanya, lakini chini ya uwezo wa Kenya wa 5–6%. Uchumi unasonga mbele, "
                f"lakini si kwa kasi ya kutosha kufyonza wafanyakazi wapya zaidi ya 800,000 wanaoingia sokoni kila mwaka. Ushindani wa kazi unaweza kuwa mkubwa zaidi."
            )
        else:
            resp = r(
                f"GDP growth is only {kf.get('gdp_growth')} — too slow for Kenya's needs. This means fewer job opportunities, "
                f"reduced government spending on public services, and cautious business investment. Approach financial decisions carefully.",
                f"Ukuaji wa GDP ni {kf.get('gdp_growth')} tu — polepole sana kwa mahitaji ya Kenya. Hii inamaanisha fursa chache za kazi, "
                f"matumizi ya serikali ya chini kwa huduma za umma, na uwekezaji wa kiangalifu wa biashara. Chukua maamuzi ya kifedha kwa uangalifu."
            )

    elif intent == "savings" and cbr is not None and inf is not None:
        rr = cbr - inf
        if rr > 2:
            resp = r(
                f"Good news — saving is actually rewarding right now! With CBK rate at {kf.get('cbr')} and inflation at {kf.get('inflation')}, "
                f"real savings returns are about {rr:.1f}%. Treasury Bills or Money Market Funds are the best options for above-inflation returns.",
                f"Habari njema — kuhifadhi akiba ni manufaa sasa hivi! Kwa kiwango cha CBK cha {kf.get('cbr')} na mfumuko wa bei wa {kf.get('inflation')}, "
                f"kurudi halisi kwa akiba ni takriban {rr:.1f}%. Treasury Bills au Mifuko ya Soko la Pesa ni chaguo bora zaidi."
            )
        elif rr >= -1:
            resp = r(
                f"Savings are barely keeping pace with inflation (real return: ~{rr:.1f}%). A standard bank savings account isn't enough. "
                f"Consider Money Market Funds or government Treasury Bills, which typically offer better inflation-adjusted returns.",
                f"Akiba zinafuata mfumuko wa bei kwa kiasi kidogo tu (kurudi halisi: ~{rr:.1f}%). Akaunti ya kawaida ya benki haitoshi. "
                f"Fikiria Mifuko ya Soko la Pesa au Treasury Bills za serikali, ambazo kwa kawaida zinatoa kurudi bora zaidi."
            )
        else:
            resp = r(
                f"Warning: inflation ({kf.get('inflation')}) is outpacing savings returns — money in a standard account loses about {abs(rr):.1f}% "
                f"in real value per year. Move savings urgently to Money Market Funds, government bonds, or other productive investments.",
                f"Tahadhari: mfumuko wa bei ({kf.get('inflation')}) unazidi kurudi kwa akiba — pesa katika akaunti ya kawaida inapoteza takriban {abs(rr):.1f}% "
                f"kwa thamani halisi kwa mwaka. Hamia haraka akiba yako kwenye Mifuko ya Soko la Pesa, dhamana za serikali, au uwekezaji mwingine."
            )

    else:
        if search_results and search_results[0]["score"] >= 0.25:
            best = search_results[0]["sentence"]
            resp = r(
                f'From the CBK report: "{best}"\n\nThis reflects CBK\'s ongoing work to maintain price stability and support sustainable economic growth.',
                f'Kutoka ripoti ya CBK: "{best}"\n\nHii inaonyesha juhudi za CBK za kudumisha utulivu wa bei na kukuza ukuaji endelevu wa uchumi.'
            )
        else:
            resp = r(
                "I couldn't find a specific answer in the loaded report. Try rephrasing your question, or use the Search tab for deeper exploration.",
                "Sijapata jibu sahihi katika ripoti. Jaribu kuuliza tena kwa njia tofauti, au tumia kichupo cha 'Tafuta' kwa utafutaji wa kina zaidi."
            )

    return {"answer": resp, "type": intent}

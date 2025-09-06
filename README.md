## Hi there 👋

<!--
**pranu-1410/pranu-1410** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
--># Comprehensive notebook-style code to perform:
# - EDA
# - Preprocessing
# - TF-IDF feature extraction
# - Supervised classification (if labels present)
# - Unsupervised clustering / topic modeling (if no labels)
# - Simple lexicon-based sentiment analysis
# - Basic visualizations (matplotlib) and saved artifacts
# This will run on the uploaded CSV at /mnt/data/68b1acd44f393_Sample_Support_Emails_Dataset (1).csv
# The outputs (plots and model files) will be saved under /mnt/data and links provided at the end.
# Note: This environment may not have all NLP corpora; to keep it robust we use built-in/simple lists where needed.

import os
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt

# helper to display dataframe in the UI (provided by the notebook environment)
from caas_jupyter_tools import display_dataframe_to_user

DATA_PATH = "/mnt/data/68b1acd44f393_Sample_Support_Emails_Dataset (1).csv"
OUT_DIR = "/mnt/data/email_nlp_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load dataset
df = pd.read_csv(DATA_PATH)
print("Loaded dataset with shape:", df.shape)

# Quick peek
display_dataframe_to_user("Sample of uploaded emails", df.head(10))

# 2) Basic EDA
print("\nColumns found:", list(df.columns))
print("\nMissing values per column:")
print(df.isnull().sum())

# Try to find likely text column and label/date columns
possible_text_cols = [c for c in df.columns if 'subject' in c.lower() or 'body' in c.lower() or 'message' in c.lower() or 'email' in c.lower() or 'text' in c.lower()]
possible_label_cols = [c for c in df.columns if 'label' in c.lower() or 'category' in c.lower() or 'intent' in c.lower() or 'folder' in c.lower() or 'class' in c.lower()]
possible_date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

print("\nDetected text columns candidate(s):", possible_text_cols)
print("Detected label columns candidate(s):", possible_label_cols)
print("Detected date columns candidate(s):", possible_date_cols)

# Heuristic: prefer 'message' > 'body' > 'text' > 'subject'
def pick_text_column(cols):
    priority = ['message','body','text','email','subject','content','description','msg']
    for p in priority:
        for c in cols:
            if p in c.lower():
                return c
    return None

TEXT_COL = pick_text_column(possible_text_cols) or (df.columns[0] if df.shape[1] == 1 else None)
LABEL_COL = possible_label_cols[0] if possible_label_cols else None
DATE_COL = possible_date_cols[0] if possible_date_cols else None

print("\nChosen TEXT_COL:", TEXT_COL)
print("Chosen LABEL_COL:", LABEL_COL)
print("Chosen DATE_COL:", DATE_COL)

if TEXT_COL is None:
    # fallback: find the longest text column by average length
    lengths = {c: df[c].astype(str).map(len).mean() for c in df.columns if df[c].dtype == object}
    if lengths:
        TEXT_COL = max(lengths, key=lengths.get)
        print("Fallback TEXT_COL chosen:", TEXT_COL)
    else:
        raise ValueError("No text-like column found in the dataset. Please provide a dataset with at least one text column.")

# 3) Preprocessing utilities
simple_stopwords = set("""a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves""".split())
def preprocess_text(s):
    if pd.isnull(s):
        return ""
    s = str(s)
    s = s.lower()
    # remove urls and emails
    s = re.sub(r'\S+@\S+', ' ', s)
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    # remove punctuation
    s = s.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # remove short tokens and stopwords
    tokens = [t for t in s.split() if len(t) > 2 and t not in simple_stopwords and not t.isdigit()]
    return " ".join(tokens)

# Apply preprocessing
df['_clean_text'] = df[TEXT_COL].apply(preprocess_text)
display_dataframe_to_user("Cleaned text sample", df[[TEXT_COL, '_clean_text']].head(10))

# 4) Simple lexicon-based sentiment analyzer (small lexicon)
positive_words = {"good","great","thank","thanks","resolved","happy","helpful","awesome","excellent","fine","resolved","appreciate","nice","love","works","working","successful"}
negative_words = {"not","bad","unable","fail","failed","issue","problem","error","delay","angry","frustrat","frustration","sad","wrong","refund","complain","complaint","cancel","slow"}

def lexicon_sentiment(text):
    if not text:
        return 0.0
    tokens = text.split()
    pos = sum(1 for t in tokens if any(p == t or t.startswith(p) for p in positive_words))
    neg = sum(1 for t in tokens if any(n == t or t.startswith(n) for n in negative_words))
    if pos+neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

df['_sentiment_score'] = df['_clean_text'].apply(lexicon_sentiment)
# label sentiment buckets
def sentiment_label(score):
    if score > 0.2:
        return 'positive'
    elif score < -0.2:
        return 'negative'
    else:
        return 'neutral'
df['_sentiment'] = df['_sentiment_score'].apply(sentiment_label)

print("\nSentiment value counts:")
print(df['_sentiment'].value_counts())

# Save sentiment summary
df[[' _sentiment' if '_sentiment' in df.columns else '_sentiment','_sentiment_score']].head(5) if '_sentiment' in df.columns else None

# 5) If label exists, prepare supervised classification. Otherwise do clustering + topic modeling
results = {}
if LABEL_COL:
    print("\n--- Supervised classification path ---")
    # Drop rows with missing labels
    df2 = df[[TEXT_COL, '_clean_text', LABEL_COL]].dropna(subset=[LABEL_COL])
    X = df2['_clean_text']
    y = df2[LABEL_COL].astype(str)
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    print("Training classifier...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy on test set:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    # save model
    model_path = os.path.join(OUT_DIR, "email_classifier_pipeline.pkl")
    joblib.dump(pipe, model_path)
    print("Saved classifier to:", model_path)
    results['classifier'] = model_path
    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    fig, ax = plt.subplots(figsize=(6,5))
    ax.matshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(np.unique(y))))
    ax.set_xticklabels(np.unique(y), rotation=45, ha='right')
    ax.set_yticks(range(len(np.unique(y))))
    ax.set_yticklabels(np.unique(y))
    plt.tight_layout()
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig)
    results['confusion_matrix'] = cm_path
else:
    print("\n--- Unsupervised path: clustering and topic modeling ---")
    # Use TF-IDF then KMeans clustering and LDA to surface topics
    X = df['_clean_text'].fillna("")
    # vectorize
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(X)
    # KMeans clustering
    n_clusters = min(6, max(2, int(len(df)/50)))  # heuristic
    print("Clustering into", n_clusters, "clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_tfidf)
    df['_cluster'] = clusters
    # Save clustering labels
    cluster_path = os.path.join(OUT_DIR, "clusters_sample.csv")
    df[[TEXT_COL, '_clean_text', '_cluster', '_sentiment']].to_csv(cluster_path, index=False)
    print("Saved cluster sample to:", cluster_path)
    results['clusters_csv'] = cluster_path
    # LDA topic modeling (use CountVectorizer for LDA)
    count_vect = CountVectorizer(max_features=10000, ngram_range=(1,2))
    X_counts = count_vect.fit_transform(X)
    n_topics = min(6, max(2, int(len(df)/100)))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_counts)
    # get top words per topic
    def show_topics(model, feature_names, n_top_words):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            topics.append((topic_idx, top_features))
        return topics
    topics = show_topics(lda, count_vect.get_feature_names_out(), 12)
    print("\nTop words per topic:")
    for tid, words in topics:
        print("Topic", tid, ":", ", ".join(words))
    # save topic listing
    topics_path = os.path.join(OUT_DIR, "lda_topics.txt")
    with open(topics_path, "w") as f:
        for tid, words in topics:
            f.write(f"Topic {tid}: {', '.join(words)}\n")
    results['lda_topics'] = topics_path

# 6) Visualizations
print("\nGenerating visualizations...")

# Sentiment distribution bar chart
sent_counts = df['_sentiment'].value_counts()
fig, ax = plt.subplots()
ax.bar(sent_counts.index.astype(str), sent_counts.values)
ax.set_title("Sentiment distribution")
ax.set_ylabel("Count")
plt.tight_layout()
sent_path = os.path.join(OUT_DIR, "sentiment_distribution.png")
plt.savefig(sent_path)
plt.close(fig)
results['sentiment_plot'] = sent_path

# Top terms (TF-IDF average)
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,1))
X_vec = vectorizer.fit_transform(df['_clean_text'].fillna(""))
tfidf_means = np.asarray(X_vec.mean(axis=0)).ravel()
top_n = 25
top_idx = tfidf_means.argsort()[::-1][:top_n]
top_terms = [(vectorizer.get_feature_names_out()[i], tfidf_means[i]) for i in top_idx]
top_terms_df = pd.DataFrame(top_terms, columns=['term','avg_tfidf'])
display_dataframe_to_user("Top TF-IDF terms", top_terms_df.head(25))

# plot top terms
fig, ax = plt.subplots(figsize=(8,5))
ax.barh(range(len(top_terms_df)), top_terms_df['avg_tfidf'].values[::-1])
ax.set_yticks(range(len(top_terms_df)))
ax.set_yticklabels(top_terms_df['term'].values[::-1])
ax.set_title("Top TF-IDF terms (top 25)")
plt.tight_layout()
terms_path = os.path.join(OUT_DIR, "top_tfidf_terms.png")
plt.savefig(terms_path)
plt.close(fig)
results['top_terms_plot'] = terms_path

# If there's a DATE_COL try to plot email volume over time
if DATE_COL and DATE_COL in df.columns:
    try:
        df['_date_parsed'] = pd.to_datetime(df[DATE_COL], errors='coerce')
        vol = df.set_index('_date_parsed').resample('D').size().dropna()
        if len(vol) > 1:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(vol.index, vol.values)
            ax.set_title("Email volume over time (daily)")
            ax.set_ylabel("Count")
            plt.tight_layout()
            vol_path = os.path.join(OUT_DIR, "email_volume_time.png")
            plt.savefig(vol_path)
            plt.close(fig)
            results['volume_plot'] = vol_path
    except Exception as e:
        print("Could not parse dates:", e)

# 7) Save the processed dataset and basic report
processed_path = os.path.join(OUT_DIR, "processed_emails_sample.csv")
df.to_csv(processed_path, index=False)
results['processed_csv'] = processed_path
print("\nSaved processed dataset to:", processed_path)

# 8) Summary of artifacts saved
print("\nArtifacts saved under:", OUT_DIR)
for k,v in results.items():
    print(k, "->", v)

# Provide user-friendly download links
print("\nDownloadable files (paths you can access):")
for name, path in results.items():
    print(f"{name}: file://{path}")

# Show a small sample with key columns
sample_cols = [TEXT_COL, '_clean_text', '_sentiment']
if LABEL_COL:
    sample_cols.append(LABEL_COL)
if '_cluster' in df.columns:
    sample_cols.append('_cluster')

display_dataframe_to_user("Final sample with key fields", df[sample_cols].head(20))

# Save model (if supervised) and vectorizers already saved above. End.
print("\nDone. Check the /mnt/data/email_nlp_outputs folder for plots, models, and csvs.")


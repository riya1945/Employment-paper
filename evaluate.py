"""
================================================================================
Automated Resume Classification: End-to-End Benchmark Evaluation Script
================================================================================
Author: Sanjeet S. Giri et al.
Paper: Early Fusion of Dual-Transformer Embeddings with an XGBoost Meta-Learner
       for Automated Resume Classification (IEEE Access)

This script executes the complete evaluation benchmark across all models:
- Phase I Unsupervised Lexical Retrieval (TF-IDF + Cosine)
- Phase I Unsupervised Dense Semantic Retrieval (MPNet Centroids)
- Phase I Unsupervised Linear Hybrid Retrieval (alpha = 0.90)
- Supervised Lexical Baseline: XGBoost (TF-IDF Features)
- Supervised Contextual Deep Learning: DistilBERT [CLS]
- Supervised Dense Semantic: MPNet (Single-Transformer Ablation)
- Alternative Fusion: Late Fusion (Soft Probability Averaging)
- Alternative Meta-Learner: Linear Meta-Learner (Logistic Regression)
- Alternative Meta-Learner: Random Forest (1536-D)
- Proposed Architecture: Early Fusion Dual-Transformer + XGBoost Meta-Learner (1536-D)
- Statistical Significance: McNemar's Paired Chi-Square & Exact Binomial Test
================================================================================
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import torch

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    if not os.path.exists("clean_resumes.csv") or not os.path.exists("clean_job_descriptions.csv"):
        print("Harmonized datasets not found locally. Running prepare_data.py...")
        import prepare_data
        prepare_data.main()
        
    resumes_df = pd.read_csv("clean_resumes.csv")
    resumes_df["Category"] = resumes_df["Category"].str.strip().str.upper()
    
    job_df = pd.read_csv("clean_job_descriptions.csv")
    job_df.columns = job_df.columns.str.lower()
    
    return resumes_df, job_df

def compute_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    correct = int(np.sum(y_true == y_pred))
    total = len(y_true)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        "Paradigm / Model": model_name,
        "Correct / 376": f"{correct} / {total}",
        "Accuracy": acc,
        "Macro F1": macro_f1
    }

def run_benchmark():
    print("=" * 80)
    print("RESUME CLASSIFICATION MASTER BENCHMARK (IEEE ACCESS REPRODUCTION)")
    print("=" * 80)
    
    resumes_df, job_df = load_data()
    categories = sorted(resumes_df["Category"].unique())
    num_classes = len(categories)
    
    le = LabelEncoder()
    resumes_df["label"] = le.fit_transform(resumes_df["Category"])
    
    train_df, test_df = train_test_split(
        resumes_df,
        test_size=0.20,
        stratify=resumes_df["label"],
        random_state=SEED
    )
    
    train_texts = train_df["Resume_str"].tolist()
    test_texts = test_df["Resume_str"].tolist()
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    
    results = []
    
    # --------------------------------------------------------------------------
    # 1. PHASE I: TF-IDF + Cosine Retrieval Baseline
    # --------------------------------------------------------------------------
    print("\n[1/6] Evaluating Phase I: Lexical Retrieval (TF-IDF + Cosine)...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    job_tfidf = tfidf.fit_transform(job_df["descriptions"])
    test_tfidf = tfidf.transform(test_texts)
    
    # Compute job centroids per mapped category
    job_df_mapped = job_df.dropna(subset=["mapped_category"]).copy()
    job_centroids = []
    for cat in categories:
        cat_mask = (job_df_mapped["mapped_category"].str.upper() == cat).values
        if np.any(cat_mask):
            centroid = np.asarray(job_tfidf[cat_mask].mean(axis=0))
        else:
            centroid = np.zeros((1, job_tfidf.shape[1]))
        job_centroids.append(centroid.flatten())
    job_centroids = np.array(job_centroids)
    
    lexical_sim = cosine_similarity(test_tfidf, job_centroids)
    y_pred_tfidf = np.argmax(lexical_sim, axis=1)
    results.append(compute_metrics(y_test, y_pred_tfidf, "Lexical IR: TF-IDF + Cosine Similarity"))
    
    # --------------------------------------------------------------------------
    # 2. PHASE I: Dense Semantic Matching (MPNet Centroids)
    # --------------------------------------------------------------------------
    print("[2/6] Evaluating Phase I: Dense Semantic Retrieval (MPNet Centroids)...")
    from sentence_transformers import SentenceTransformer
    mpnet = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    # Encode test resumes
    mpnet_test_emb = mpnet.encode(test_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    mpnet_train_emb = mpnet.encode(train_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    
    # Pre-computed or sample centroid matching
    # Compute semantic centroids from training resumes or job postings
    semantic_centroids = []
    for c_idx in range(num_classes):
        cat_emb = mpnet_train_emb[y_train == c_idx]
        semantic_centroids.append(cat_emb.mean(axis=0))
    semantic_centroids = np.array(semantic_centroids)
    semantic_centroids = semantic_centroids / np.linalg.norm(semantic_centroids, axis=1, keepdims=True)
    
    semantic_sim = cosine_similarity(mpnet_test_emb, semantic_centroids)
    y_pred_mpnet_ir = np.argmax(semantic_sim, axis=1)
    results.append(compute_metrics(y_test, y_pred_mpnet_ir, "Semantic Retrieval: MPNet Dense Matching"))
    
    # --------------------------------------------------------------------------
    # 3. PHASE I: Unsupervised Linear Hybrid Retrieval (alpha = 0.90)
    # --------------------------------------------------------------------------
    print("[3/6] Evaluating Phase I: Unsupervised Linear Hybrid (alpha = 0.90)...")
    # Normalize lexical similarities to [0, 1]
    lex_norm = (lexical_sim - lexical_sim.min()) / (lexical_sim.max() - lexical_sim.min() + 1e-8)
    sem_norm = (semantic_sim - semantic_sim.min()) / (semantic_sim.max() - semantic_sim.min() + 1e-8)
    hybrid_sim = 0.90 * sem_norm + 0.10 * lex_norm
    y_pred_hybrid = np.argmax(hybrid_sim, axis=1)
    results.append(compute_metrics(y_test, y_pred_hybrid, "Hybrid Retrieval: Unsupervised Hybrid (alpha=0.90)"))
    
    # --------------------------------------------------------------------------
    # 4. PHASE II: Supervised XGBoost Baseline (TF-IDF Features)
    # --------------------------------------------------------------------------
    print("[4/6] Evaluating Phase II: Supervised XGBoost (TF-IDF Features)...")
    tfidf_sup = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_sup.fit_transform(train_texts)
    X_test_tfidf = tfidf_sup.transform(test_texts)
    
    xgb_tfidf = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    xgb_tfidf.fit(X_train_tfidf, y_train)
    y_pred_xgb_tfidf = xgb_tfidf.predict(X_test_tfidf)
    results.append(compute_metrics(y_test, y_pred_xgb_tfidf, "Supervised Lexical: XGBoost (TF-IDF Features)"))
    
    # --------------------------------------------------------------------------
    # 5. PHASE II: Single-Transformer Supervised Models (MPNet Only & DistilBERT)
    # --------------------------------------------------------------------------
    print("[5/6] Evaluating Phase II: Controlled Single-Transformer Baselines...")
    # XGBoost on MPNet embeddings only
    xgb_mpnet = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    xgb_mpnet.fit(mpnet_train_emb, y_train)
    y_pred_xgb_mpnet = xgb_mpnet.predict(mpnet_test_emb)
    results.append(compute_metrics(y_test, y_pred_xgb_mpnet, "Single Transformer: XGBoost (MPNet Embeddings Only)"))
    
    # Contextual representations (DistilBERT [CLS])
    # Extract contextual embeddings using pre-trained / fine-tuned DistilBERT
    print("      Extracting DistilBERT contextual representations...")
    from transformers import AutoTokenizer, AutoModel
    db_name = "distilbert-base-uncased"
    db_tokenizer = AutoTokenizer.from_pretrained(db_name)
    db_model = AutoModel.from_pretrained(db_name).to(device)
    db_model.eval()
    
    def extract_cls(texts):
        embs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                tok = db_tokenizer(batch, max_length=256, truncation=True, padding=True, return_tensors="pt").to(device)
                out = db_model(**tok)
                cls_tok = out.last_hidden_state[:, 0, :].cpu().numpy()
                embs.append(cls_tok)
        return np.vstack(embs)
    
    db_train_emb = extract_cls(train_texts)
    db_test_emb = extract_cls(test_texts)
    
    xgb_db = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    xgb_db.fit(db_train_emb, y_train)
    y_pred_xgb_db = xgb_db.predict(db_test_emb)
    results.append(compute_metrics(y_test, y_pred_xgb_db, "Single Transformer: XGBoost (DistilBERT [CLS] Only)"))
    
    # --------------------------------------------------------------------------
    # 6. PHASE II: Proposed Early Fusion Framework (Dual-Transformer + XGBoost)
    # --------------------------------------------------------------------------
    print("[6/6] Evaluating Proposed Framework: Dual-Transformer Early Fusion (1536-D)...")
    ef_train = np.hstack([db_train_emb, mpnet_train_emb])
    ef_test = np.hstack([db_test_emb, mpnet_test_emb])
    
    # Alternative Meta-Learners
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(ef_train, y_train)
    y_pred_lr = lr.predict(ef_test)
    results.append(compute_metrics(y_test, y_pred_lr, "Alternative Meta-Learner: Linear (Logistic Regression)"))
    
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(ef_train, y_train)
    y_pred_rf = rf.predict(ef_test)
    results.append(compute_metrics(y_test, y_pred_rf, "Alternative Meta-Learner: Random Forest (1536-D)"))
    
    # Proposed Early Fusion XGBoost
    xgb_ef = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1
    )
    xgb_ef.fit(ef_train, y_train)
    y_pred_ef = xgb_ef.predict(ef_test)
    results.append(compute_metrics(y_test, y_pred_ef, "Proposed Framework: Early Fusion XGBoost (1536-D)"))
    
    # --------------------------------------------------------------------------
    # BENCHMARK REPORT TABLE
    # --------------------------------------------------------------------------
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("MASTER BENCHMARK EVALUATION RESULTS (CORRESPONDS TO MANUSCRIPT TABLE 4)")
    print("=" * 80)
    print(res_df.to_string(index=False))
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # STATISTICAL SIGNIFICANCE (MCNEMAR'S TEST)
    # --------------------------------------------------------------------------
    print("\n--- McNemar's Test: Proposed Early Fusion vs. MPNet Baseline ---")
    correct_ef = (y_pred_ef == y_test)
    correct_mpnet = (y_pred_xgb_mpnet == y_test)
    
    n00 = np.sum(correct_ef & correct_mpnet)
    n01 = np.sum(correct_ef & ~correct_mpnet)
    n10 = np.sum(~correct_ef & correct_mpnet)
    n11 = np.sum(~correct_ef & ~correct_mpnet)
    
    chi2 = (abs(n01 - n10) ** 2) / (n01 + n10)
    chi2_yates = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    p_val = stats.chi2.sf(chi2, df=1)
    p_yates = stats.chi2.sf(chi2_yates, df=1)
    p_binom = stats.binomtest(n01, n01 + n10, 0.5, alternative='two-sided').pvalue
    
    print(f"Contingency Matrix (N={len(y_test)}):")
    print(f"  n00 (Both Correct):    {n00}")
    print(f"  n01 (Proposed Only):   {n01}")
    print(f"  n10 (MPNet Only):      {n10}")
    print(f"  n11 (Both Incorrect):  {n11}")
    print(f"Discordant Pairs:        {n01 + n10} (Ratio: {n01}:{n10})")
    print(f"Asymptotic Chi-Square:   chi2 = {chi2:.3f}, p = {p_val:.4f}")
    print(f"Yates Corrected:         chi2 = {chi2_yates:.3f}, p = {p_yates:.4f}")
    print(f"Exact Binomial Test:     p = {p_binom:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()


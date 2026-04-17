# Improving Recommendation Accuracy: Audio-Based Music Recommendation using Local Sonic Analysis (AudioMuse-AI) and Last.fm Top Tags
**CSE 575 Final Project Report**

## Team Members
* **Wazir Khan (Person 1 / Khush Manchanda)**: Dataset setup, pipelining, baseline algorithms.
* **Khush Manchanda (Person 2 / Arjun Ranjan)**: Last.fm Tag pipeline.
* **Arjun Ranjan (Person 3 / Diggy)**: Audio feature extraction (Librosa).
* **Shashank Valayaputtur (Person 4 / Ninjaman)**: Modeling and evaluation framework (CF, MF, hybrid).
* **Deeparghya Dutta Barua (Person 5)**: Evaluation criteria, report writing, and plots.

---

## 1. Introduction and Problem Statement
Traditional collaborative filtering (CF) algorithms recommend items by measuring the overlapping interactions of users. However, in music recommendation, CF suffers heavily from inherent sparsity (the "cold-start" problem) where new or unpopular songs have few ratings, and consequently, recommendations degrade. In reality, two songs might be highly similar based on content, such as their **sonic profile** (timbre, tempo) or **semantic description** (tags). 

**Research Question**: How much do audio-derived features and Last.fm tags improve rating prediction and top-N recommendation quality, especially for sparse users and items (cold-start or near-cold-start cases)?

To answer this, we built **AudioMuse-AI**, a hybrid music recommendation system leveraging explicit ratings (collaborative signals), Last.fm community tags (semantic signals), and Librosa-extracted MP3 summaries (audio signals) and evaluated their ablations across different sparsity regimes.

---

## 2. Methodology & Features

### 2.1 Collaborative Ratings (Person 1)
We utilized the **HetRec 2011 Last.fm 2K** dataset to construct our core explicitly-rated dataset. We formatted the raw scale into a normalized continuous explicitly-fed space ranging from [1.0 to 5.0].
Our pre-processing yields:
- 1,892 Training Users and 1,884 Test Users 
- ~92.8K total rating tuples, forming an extremely sparse 0.26% rating density matrix matrix.
- We implement models across varying data richness to specifically study sparse limits (Cold users ≤5 ratings, Near-Cold 6-20 ratings, Normal >20 ratings).

### 2.2 Semantic Tags (Person 2)
We fetched semantic characteristics of tracks via Last.fm top tags (genres, moods). We normalized unstructured tags (e.g., lower casing, synonym conflation) down to the globally most frequent terms. We constructed **TF-IDF vectors** (100 dimensions) to concisely map an item into a normalized semantic continuous space which inherently weighs discriminative tags over generic descriptors.

### 2.3 Audio Sonic Features (Person 3)
We mapped MusicNet classical audio equivalents to HetRec artists (handling naming ambiguities and matching strings) to ground classical tracks with genuine `.wav` audio signals. We apply `librosa` over the audio snippets, aggregating 61 low-level summary acoustic indicators per item including: MFCCs, chroma feature distributions, spectral contrast, and RMS energy statistics.

---

## 3. Evaluated Models (Person 4)

We train sequential ablations to measure content value:
1. **Baselines (Global / User / Item Mean)**: Non-personalized dataset aggregates.
2. **User-kNN**: Standard User-based collaborative filtering utilizing cosine similarity over the rated item vectors.
3. **Matrix Factorization (MF)**: Truncated SVD reducing the raw item-user matrix into $k=50$ dense latent features. Only user-ratings are given.
4. **MF + Tags (Hybrid)**: Content-infused MF where normalized tag TF-IDF features modulate the item factors. 
5. **MF + Audio (Hybrid)**: Content-infused MF where scaled acoustic descriptors modulate the item factors.
6. **MF + Tags + Audio**: The unified full-hybrid model.

---

## 4. Evaluation and Results (Person 5)

We evaluated performance on two fronts: 
**A. Rating Prediction** (RMSE, MAE over explicit rating values)
**B. Ranking Discoverability** (Precision@10, Recall@10, NDCG@10 for held out 3.5+ rated interactions)

### 4.1 General Results
| Model | RMSE | MAE | P@10 | R@10 | NDCG@10 |
|-------|------|-----|------|------|----------|
| Global Mean | 0.4851 | 0.3679 | 0.0000 | 0.0000 | 0.0000 |
| User Mean | **0.2588** | **0.1954** | 0.0000 | 0.0000 | 0.0000 |
| Item Mean | 0.5043 | 0.3831 | 0.0000 | 0.0000 | 0.0000 |
| **kNN (user-based)** | *0.2750* | *0.2072* | **0.0304** | **0.2232** | **0.1405** |
| MF (ratings only) | 0.3711 | 0.2749 | 0.0000 | 0.0000 | 0.0000 |
| MF + tags | 0.4462 | 0.3373 | 0.0008 | 0.0035 | 0.0030 |
| MF + audio | 0.4441 | 0.3356 | 0.0013 | 0.0044 | 0.0037 |
| MF + full | 0.4481 | 0.3385 | 0.0013 | 0.0044 | 0.0037 |

**Key Finding**: Pure User Mean strictly dominates RMSE, suggesting Last.fm rating behavior is incredibly user-biased (users anchor rating distributions to internal standards rather than items having absolute value). kNN excels exceptionally at ranking (0.14 NDCG@10) due to localized cluster discovery, while the global MF falls short dynamically ordering small implicit lists. Content-hybrids (tags/audio) *degrade* RMSE relatively but introduce non-zero ranking capabilities for MF. 

### 4.2 Cold-Start and Sparsity Analysis

We evaluated isolated models over user cohorts: Cold (≤5 ratings), Near-cold (6-20 ratings), Normal (>20 ratings).

| Segment | Cold (n=8) | Near-Cold (n=19) | Normal (n=1857)|
|---------|---------|----------|----------|
| **Global Mean** | 0.2975 | 0.3873 | 0.2583|
| **MF (Ratings Only)** | 0.6518 | 0.6956 | 0.3696 |
| **MF + Tags** | 0.6519 | 0.7026 | 0.4457 |

**Conclusion on Cold-Start**:
Due to extreme data sparsity in true "cold" zones (only 8 validation-valid users out of 1900), the simpler models (Global bounds) significantly outperform parameterized factorization approaches which over-regularize sparse features.

---

## 5. Conclusion
We successfully engineered an end-to-end data processing and model evaluation pipeline that evaluates multi-modal information signals.
- Audio and Semantic representations function to enrich ranking diversity where basic sparse models collapse (MF P@10 0.0 $\rightarrow$ MF+Full P@10 0.0013).
- User-bias behavior is overwhelmingly dominant concerning raw scoring tasks predicting 1.0 - 5.0 explicit feedback values over Last.fm data.
- kNN architectures outclass SVD factorization in sparse binary discovery domains (R@10). 

*(Refer to `notebooks/figures/` for high-resolution distribution graphs, sparsity checks, RMSE ablations, and generated sample user playlist tables.)*

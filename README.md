# NEWSCOPE: Diverse News Retrieval

This repo contains the official implementation and datasets for the paper: **Uncovering the Bigger Picture: Comprehensive Event Understanding via Diverse News Retrieval** (Accepted at *EMNLP 2025*）

## Overview

Most news retrieval systems focus primarily on textual relevance, often surfacing redundant articles and limiting exposure to alternative viewpoints. This work introduces **NEWSCOPE**, a two-stage framework for **diverse news retrieval**, designed to surface complementary perspectives and improve event coverage.

<img width="1570" height="761" alt="2505_newscope" src="https://github.com/user-attachments/assets/e31b9809-cbf4-414a-abee-817fc7c4acf9" />

**Key Features:**

- **Two-Stage Retrieval Framework**
    1. **Stage I:** Dense relevance-based retrieval.
    2. **Stage II:** Sentence-level clustering and diversity-aware re-ranking.
- **Novel Diversity Metrics**
    - *Average Pairwise Distance (D)*
    - *Positive Cluster Coverage (C)*
    - *Information Density Ratio (I)*
- **Benchmarks**
    - **LocalNews**: 103 local events, 5,296 annotated paragraphs.
    - **DSGlobal**: 147 global events, 7,532 annotated paragraphs.

## Data

We release two paragraph-level benchmarks to support evaluation of diverse news retrieval:

1. **LocalNews**
    - Built from Google News “Full Coverage.”
    - 103 events, 5,296 paragraphs.
    - Paragraphs labeled for relevance to abstractive event summaries.
2. **DSGlobal**
    - Adapted from the *DiverseSumm* dataset.
    - 147 global events, 7,532 paragraphs.
    - Focused on evaluating generalizability across global news.

Please upzip the dataset before running.

## Installation

```bash
cd NEWSCOPE
pip install -r requirements.txt

```

## Usage

### Run Retrieval

```bash
sh scripts\do_retrieval.sh <data_root_path> <diverse_rerank_method>

```

Arguments:

- `$1`: data_root_path (`LocalNews` or `DSGlobal`)
- `$2`: diverse_rerank_method (`GreedySCS` or `GreedyPlus`)

### Evaluate

```bash
sh scripts\do_eval.sh <data_root_path> <output_fn>

```

Arguments:

- `$1`: data_root_path (`LocalNews` or `DSGlobal`)
- `$2`: output_fn (e.g. “eval_result”)

## Results

NEWSCOPE consistently outperforms strong baselines on both **LocalNews** and **DSGlobal**, achieving significantly higher **diversity** without compromising **relevance**.

- Performance on LocalNews
<img width="5756" height="2016" alt="2505_newscope_local" src="https://github.com/user-attachments/assets/6cc4e6e0-722e-407e-a278-50a54e89a1fa" />

- Performance on GlobalNews
<img width="5756" height="2016" alt="2505_newscope_global" src="https://github.com/user-attachments/assets/a8091bfe-feb8-40ad-a05e-fa98a3a6aa0d" />

*See full tables in the paper.*

## Citation

If you use our code or data, please cite:

```
@inproceedings{newscope2025,
  title={Uncovering the Bigger Picture: Comprehensive Event Understanding via Diverse News Retrieval},
  author={Tang Yixuan and Shi Yuanyuan and Sun Yiqun and Anthongy K.H. Tung},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025}
}
```

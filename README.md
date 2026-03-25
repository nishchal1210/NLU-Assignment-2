# NLU Assignment-2

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![IIT Jodhpur](https://img.shields.io/badge/IIT-Jodhpur-orange.svg)](https://iitj.ac.in/)

> **Natural Language Understanding** — Word Embeddings & Character-Level Name Generation
> *All models implemented from scratch using pure NumPy*

---

## Overview

This repository contains the implementation of two fundamental NLP tasks for the NLU course at IIT Jodhpur:

| Problem | Task | Models |
|---------|------|--------|
| **Problem 1** | Word Embeddings on IIT Jodhpur Corpus | CBOW, Skip-gram |
| **Problem 2** | Character-Level Name Generation | Vanilla RNN, BiLSTM, RNN+Attention |

**Key Highlight:** No PyTorch, No TensorFlow — Everything built from scratch!

---

## Repository Structure

```
NLU-Assignment-2/
│
├── iitj_scraper.ipynb           # Web scraper for IIT Jodhpur corpus
├── word2vec_iitj.ipynb          # CBOW & Skip-gram implementation
├── rnn_name_generation.ipynb    # RNN variants for name generation
├── TrainingNames.txt            # Indian names dataset (1000 names)
│
├── iitj_corpus/                 # Scraped corpus data
│   ├── all_docs.json            # Complete corpus in JSON format
│   ├── corpus.txt               # Preprocessed text corpus
│   ├── academics.txt
│   ├── BTech.txt
│   ├── centre.txt
│   ├── dept.txt
│   ├── faculty.txt
│   ├── idrp.txt
│   ├── news.txt
│   ├── notice.txt
│   ├── pdf_regulation.txt
│   └── school.txt
│
├── saved_models/                # Trained RNN models
│   ├── vanilla_rnn.pkl
│   ├── bilstm.pkl
│   └── attention_rnn.pkl
│
├── cbow_model.pkl               # Trained CBOW model
├── skipgram_model.pkl           # Trained Skip-gram model
├── cbow_embeddings_300.npz      # CBOW embeddings (NumPy format)
├── skipgram_embeddings_300.npz  # Skip-gram embeddings (NumPy format)
│
└── *.png                        # Visualizations
```

---

## Problem 1: Word Embeddings

### Dataset
Custom corpus scraped from [IIT Jodhpur website](https://iitj.ac.in/):

| Metric | Value |
|--------|-------|
| Documents | 80 |
| Total Tokens | 38,291 |
| Vocabulary Size | 4,766 |

### Models Implemented

#### CBOW (Continuous Bag of Words)
Predicts target word from surrounding context.

```
h = (1/2c) × Σ W_in[context_words]
y = softmax(W_out × h)
```

#### Skip-gram
Predicts context words from target word.

```
h = W_in[target_word]
y_i = softmax(W_out × h)  ∀ context words
```

### Results — Nearest Neighbours

**CBOW Model:**
| Query | Top-3 Similar Words |
|-------|---------------------|
| research | faculty (0.985), design (0.983), program (0.980) |
| student | semester (0.990), academic (0.989), course (0.988) |
| phd | program (0.971), students (0.964), design (0.964) |

**Skip-gram Model:**
| Query | Top-3 Similar Words |
|-------|---------------------|
| research | facility (0.628), state (0.594), art (0.584) |
| phd | mtech (0.887), admissions (0.885), postgraduate (0.878) |
| course | study (0.914), self (0.903), independent (0.892) |

### Visualizations

<p align="center">
  <img src="wordcloud_iitj.png" width="45%" alt="Word Cloud"/>
  <img src="pca_comparison.png" width="45%" alt="PCA Comparison"/>
</p>

<p align="center">
  <img src="training_loss.png" width="45%" alt="Training Loss"/>
  <img src="similarity_heatmap.png" width="45%" alt="Similarity Heatmap"/>
</p>

---

## Problem 2: Name Generation

### Dataset
Indian names dataset for character-level modeling:

| Metric | Value |
|--------|-------|
| Total Names | 1,000 |
| Unique Names | 890 |
| Vocabulary (chars) | 51 |
| Mean Length | 6.7 |

### Models Implemented

| Model | Architecture | Parameters |
|-------|--------------|------------|
| **Vanilla RNN** | `h_t = tanh(W_xh·x + W_hh·h + b)` | 29,619 |
| **BiLSTM** | Forward + Backward LSTM | 197,427 |
| **RNN + Attention** | RNN encoder + Bahdanau attention | 44,403 |

### Results

| Model | Realistic % | Diversity % | Avg Length | Final Loss |
|-------|-------------|-------------|------------|------------|
| Vanilla RNN | **99.6%** | 98.8% | 5.91 | 1.989 |
| BiLSTM | 1.2% | 66.2% | 2.01 | 0.093 |
| RNN+Attention | **99.8%** | 98.6% | 5.86 | 1.907 |

### Generated Names

**Vanilla RNN:**
```
Ghylapa, Jodin, Nishal, Aruha, Nitsha, Kananda, Somita, Saniya
```

**RNN + Attention:**
```
Thashn, Kijav, Suvrajid, Oopara, Suhan, Viduya, Radminanha, Sanandeet
```

### Key Finding
> **BiLSTM Failure:** Despite achieving the lowest training loss (0.093), BiLSTM produced only 1.2% realistic names. This demonstrates that bidirectional models are unsuitable for autoregressive generation without architectural modifications — future context isn't available during inference.

### Visualizations

<p align="center">
  <img src="rnn_comparison.png" width="45%" alt="RNN Comparison"/>
  <img src="metrics_comparison.png" width="45%" alt="Metrics Comparison"/>
</p>

<p align="center">
  <img src="length_distribution.png" width="45%" alt="Length Distribution"/>
  <img src="bigrams.png" width="45%" alt="Bigrams"/>
</p>

---

## Installation & Usage

### Prerequisites
```bash
pip install numpy matplotlib nltk scikit-learn wordcloud requests beautifulsoup4 pdfplumber
```

### Running the Notebooks

1. **Scrape IIT Jodhpur Corpus:**
   ```bash
   jupyter notebook iitj_scraper.ipynb
   ```

2. **Train Word2Vec Models:**
   ```bash
   jupyter notebook word2vec_iitj.ipynb
   ```

3. **Train RNN Name Generators:**
   ```bash
   jupyter notebook rnn_name_generation.ipynb
   ```

### Load Pre-trained Models

```python
import pickle
import numpy as np

# Load Word2Vec
with open('cbow_model.pkl', 'rb') as f:
    cbow = pickle.load(f)

# Load RNN
with open('saved_models/attention_rnn.pkl', 'rb') as f:
    rnn_attn = pickle.load(f)

# Load embeddings
embeddings = np.load('cbow_embeddings_300.npz')
```

---

## Technical Highlights

### From-Scratch Implementations
- Numerically stable **softmax** and **sigmoid** functions
- **Backpropagation Through Time (BPTT)** for RNNs
- **Gradient clipping** by global L2 norm
- **Negative sampling** for efficient Word2Vec training
- **Xavier initialization** for weight matrices
- **Bahdanau-style additive attention** mechanism

### Optimizations
- Subsampling of frequent words
- Learning rate decay scheduling
- Temperature-controlled sampling for generation

---

## References

1. Mikolov et al. (2013) — [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
2. Mikolov et al. (2013) — [Distributed Representations of Words and Phrases](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality)
3. Graves (2013) — [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)
4. Bahdanau et al. (2014) — [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
5. Hochreiter & Schmidhuber (1997) — [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

## Author

**Nishchal**
B22CS042
Department of Computer Science & Engineering
Indian Institute of Technology Jodhpur

---

## License

This project is for educational purposes as part of the NLU course at IIT Jodhpur.

---

<p align="center">
  <i>Made with NumPy and determination</i>
</p>

# Enhanced Resume Screening Using LLMs and RAG Pipelines

This repository contains the implementation of the MSc dissertation project:

**Enhanced Resume Screening Using Large Language Models with Retrieval-Augmented Generation (RAG) Pipelines**

**Author:** Mai Ali Abdel Qader  
**Degree:** MSc Artificial Intelligence  
**Institution:** University of Bath (2025)

---

## 📌 Project Overview

This project evaluates the impact of structured query decomposition and advanced retrieval strategies on automated resume screening within a Retrieval-Augmented Generation (RAG) framework.

Two pipelines are implemented under identical experimental conditions:

- **Basic RAG Pipeline** – single-query semantic retrieval  
- **Advanced RAG Pipeline** – multi-subquery retrieval with heuristic optimisation, hybrid dense–sparse retrieval, and score-aware fusion

All outputs are generated using open-source LLaMA models and are strictly grounded in retrieved resume evidence.

---

## 🧠 Key Contributions

- Structured job-description subquery decomposition
- Multi-objective heuristic subquery scoring
- Hybrid dense–sparse retrieval (FAISS + TF-IDF)
- Score-aware Reciprocal Rank Fusion (RRF)
- **Data-driven dense–sparse fusion using logistic regression**
- Evidence-grounded LLM-based candidate ranking
- Quantitative evaluation using RAGAS

---

## 🏗️ Repository Structure

```
resume-screening/
├── requirements.txt
├── .env
├── rag_pipeline/    # Core RAG implementation
├── vectorstore/     # FAISS index
├── models/          # Fusion models
└── data/            # Resumes, test sets, and outputs
```

---

## 🔍 Pipeline Comparison

### Basic RAG Pipeline
- Treats the job description as a single holistic query
- Dense semantic retrieval using FAISS
- LLaMA ranks candidates with evidence-based justification
- Serves as a controlled baseline

### Advanced RAG Pipeline
- Decomposes job descriptions into 3–5 focused subqueries
- Subqueries scored using coverage, redundancy penalty, specificity, and topical diversity
- Each subquery retrieves resumes independently
- Results merged using score-aware RRF
- Hybrid retrieval combines dense (FAISS) and sparse (TF-IDF) similarity
- **Learned dense–sparse fusion applied at retrieval time**

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Prerequisites

Before running the pipeline, ensure the following environment variables are available:

- `HUGGINGFACE_TOKEN` is required to load LLaMA models from Hugging Face.
- `OPENAI_API_KEY` is required for RAGAS-based evaluation.

Both variables must be set prior to execution.

---

### Selecting Pipeline Mode (Basic vs Advanced)

In the main entry script, select the retrieval mode:

```python
rag_mode = "basic"      # or "advanced"
```

---

### Selecting the LLaMA Model (LLaMA 3.1 or LLaMA 2)

The pipeline supports running with different LLaMA models.  
The model is selected via environment configuration and passed at runtime.

```python
# Use LLaMA 3.1 (default)
model_name = llama3_8b_model

# Switch to LLaMA 2 if required
# model_name = llama2_13b_model
```

---

Run the main entry script from the project root:

```bash
python rag_pipeline/main.py
```

---

## 🧪 Data-Driven Fusion

The Advanced RAG pipeline uses a learned dense–sparse fusion model (logistic regression)
to combine semantic and lexical retrieval signals.

A pre-trained fusion model is already included and is automatically loaded at runtime
when running the Advanced pipeline.

Re-training the fusion model is only required if you want to regenerate it
(e.g., after modifying the training data, retrieval features, or fusion logic).

To re-train the fusion model:

```bash
python rag_pipeline/train_fusion_model.py
```

---

## 📊 Evaluation

System outputs are evaluated using RAGAS metrics:
- Context Precision
- Context Recall
- Faithfulness
- Answer Similarity

Evaluation results are exported as CSV files for analysis.

---

## 📜 License

Released for academic and research purposes only.

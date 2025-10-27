---
title: Federated RAG (FedRAG)
tags: [fedrag, llm]
dataset: [PubMed, StatPearls, Textbooks, Wikipedia, PubMedQA, BioASQ]
framework: [FAISS, transformers]
---

# Federated Retrieval Augmented Generation (FedRAG)

Federated RAG lets you query distributed corpora without centralizing data. Clients retrieve locally; the server merges and (optionally) re-ranks results, then calls an LLM.

## Quick Start

1) Install and set API key
```bash
pip install -e .
export OPENAI_API_KEY="your-api-key"
```

2) Download data and build indices
```bash
./data/prepare.sh
```

3) Ask questions
- Create `question.json` (single or list), e.g.:
```json
{ "question": "What are the symptoms of hypertension?", "title": "Internal Medicine" }
```
or use `question.txt` (one per line). Then run:
```bash
flwr run .
```

## Federated Reranker (optional)
Train a tiny linear reranker across clients; the server aggregates weights and uses them to re-rank.

1) Prepare training payload
```bash
cat > /tmp/train_payload.json << 'JSON'
{ "weights": [0.7,0.3,0.0],
  "pairs": [
    {"doc_score_norm": 0.92, "title": "Pathology_Robbins", "question_title": "Pathology", "label": 1},
    {"doc_score_norm": 0.35, "title": "Anatomy_Gray",   "question_title": "Pathology", "label": 0}
  ]
}
JSON
```

2) Run with env vars
```bash
export FEDRAG_RERANKER_TRAIN_JSON=/tmp/train_payload.json
export FEDRAG_RERANKER_WEIGHTS=/tmp/reranker_weights.json
flwr run .
```
Subsequent runs will reuse `FEDRAG_RERANKER_WEIGHTS` automatically. Falls back to heuristic or score-only.

## Corpus Splitting (5 clients)
Create five textbook parts and indices:
```bash
python3 split_corpus.py
```

## Config (pyproject.toml)
```toml
clients-corpus-names = "textbooks_part1|textbooks_part2|textbooks_part3|textbooks_part4|textbooks_part5"
k-nn = 8
k-rrf = 60
server-llm-model = "gpt-4"
```

## Troubleshooting
- Download/index: `./data/prepare.sh`
- Split corpus: `python3 split_corpus.py`
- Self-test: `python3 test_federated_setup.py`

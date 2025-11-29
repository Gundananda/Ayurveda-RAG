# 🌿 AyurGenixAI: RAG Q&A for Ayurvedic Guidance (FAISS + Sentence‑BERT + Zephyr‑7B)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-all--MiniLM--L6--v2-0A7EBB)](https://www.sbert.net/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-00A37A)](https://faiss.ai/)
[![Transformers](https://img.shields.io/badge/HF-Transformers-ff69b4?logo=huggingface&logoColor=white)](https://huggingface.co/transformers/)
[![Model](https://img.shields.io/badge/LLM-Zephyr--7B--Alpha-orange)](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A retrieval‑augmented generation (RAG) system that answers Ayurvedic questions by retrieving semantically similar entries from a curated CSV and prompting an instruction‑tuned LLM.

</div>

---

## 📌 Overview

AyurGenixAI turns a structured Ayurveda CSV into a searchable knowledge base:
- Ingests rows (Disease, Hindi Name, Symptoms, Diet, Age/Gender, Herbs, Formulations, Yoga, Recommendations)
- Encodes rows with Sentence‑BERT (all‑MiniLM‑L6‑v2)
- Indexes vectors in FAISS for fast top‑k retrieval
- Builds an instruction prompt for Zephyr‑7B‑Alpha
- Generates answers grounded in retrieved context

Important: This project is for research and education only. It is not medical advice.

---

## ✨ Key Features

- Sentence‑BERT embeddings for robust semantic retrieval
- FAISS L2 index with configurable top_k
- Instruction‑style prompt with retrieved context
- Works on GPU (preferred) or CPU (smaller LLM recommended)
- Simple, extensible code for swapping models or prompt templates

---

## 📂 Project Structure

```plaintext
ayurgenixai-rag/
├── notebook.ipynb                  # End-to-end RAG workflow
├── AyurGenixAI_Dataset.csv         # Knowledge CSV (not included)
├── README.md
└── LICENSE
```

---

## 📦 Dataset

- Path used: /content/AyurGenixAI_Dataset.csv
- Expected columns (example):
  - Disease, Hindi Name, Symptoms, Dietary Habits, Age Group, Gender,
    Herbal/Alternative Remedies, Ayurvedic Herbs, Formulation,
    Yoga & Physical Therapy, Patient Recommendations
- Each row is combined into a single “document” string for embedding.

You can add/rename fields—just update the combine_fields() function accordingly.

---

## 🧠 Technical Details

- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Index: FAISS IndexFlatL2
- Generator: HuggingFaceH4/zephyr-7b-alpha (instruction-tuned)
- Prompt template:
  - [INST] Context: <retrieved_docs> Question: <user> Answer: [/INST]
- Defaults:
  - top_k = 3
  - max_new_tokens = 300
  - temperature = 0.7

GPU recommended for Zephyr‑7B. For CPU or low VRAM, consider a smaller instruct model (e.g., Qwen2.5‑1.5B‑Instruct, Google Gemma‑2B‑it) or 4‑bit loading.

---

## 🚀 Getting Started

### Installation
```bash
pip install sentence-transformers faiss-cpu transformers accelerate torch --upgrade
# If you have CUDA and want FAISS GPU:
# pip install faiss-gpu
```

Optional 4‑bit quantization (saves VRAM):
```bash
pip install bitsandbytes
```

### Minimal Usage (inside the notebook/script)
- Load CSV → combine fields → embed → build FAISS index
- Load Zephyr‑7B → create pipeline
- Ask questions via rag_query()

Example:
```python
print(rag_query("What is an Ayurvedic treatment for joint pain in elderly women?"))
print(rag_query("What is a good herbal remedy for fever in children?"))
```

---

## ⚙️ Configuration

- Change the LLM:
```python
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # smaller, faster
```

- Load 4‑bit (if using bitsandbytes):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
```

- Save/Load FAISS index:
```python
faiss.write_index(index, "ayurgenix.index")
# ...
index = faiss.read_index("ayurgenix.index")
```

- Adjust retrieval depth:
```python
rag_query("...", top_k=5)
```

---

## 🧪 Evaluation (suggested)

- Retrieval: measure Recall@k using a set of (question, relevant_row_ids)
- Generation: human review or automatic checks for grounding (does the answer cite retrieved fields?)
- Safety: ensure medical caveats are present; prefer conservative language

Skeleton:
```python
def recall_at_k(questions, gold_ids, k=3):
    hits = 0
    for q, gold in zip(questions, gold_ids):
        qv = embed_model.encode([q])
        _, I = index.search(qv, k)
        if any(g in I[0] for g in gold): hits += 1
    return hits / len(questions)
```

---

## 🔒 Safety & Responsibility

- Outputs may be incorrect, incomplete, or unsafe.
- Always include a medical disclaimer and advise consulting qualified professionals.
- Avoid providing dosages or personalized treatment without a clinician.
- Keep dataset sources vetted; ensure consent and proper licensing.

---

## 🧩 Next Steps

- Add citations: return the IDs/fields for each retrieved chunk in the answer.
- Chunking: split long rows/sections and store chunk metadata.
- Reranking: use a cross‑encoder (e.g., ms-marco‑MiniLM‑L‑6‑v2) after FAISS retrieval.
- Caching: memoize embeddings and responses for speed.
- Serve as an API or Streamlit demo.

---

## 📄 License

Released under the MIT License. See LICENSE.

---



⭐️ If this project helps you, a star would be awesome!

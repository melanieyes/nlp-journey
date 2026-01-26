# NLP Lab 1: Machine Translation Evaluation Metrics

Hands-on implementation of standard MT/NLG evaluation metrics using real Vietnamese-English translation examples.

## 📊 Implemented Metrics

| Metric | Library | Key Features |
|--------|---------|--------------|
| **Cosine Similarity** | SentenceTransformers (`all-MiniLM-L6-v2`) | Semantic embedding similarity [0.9866] |
| **Semantic Similarity** | SentenceTransformers + `util.cos_sim` | Direct cosine on pooled embeddings [0.9866] |
| **BERTScore** | `bertscore` (RoBERTa-large) | Token-level contextual similarity<br>P: 0.8344, R: 0.6359, F1: 0.7217 |
| **ROUGE** | `rouge-score` + `rouge` | N-gram overlap (1/2/L)<br>ROUGE-1 F1: 0.8333 |
| **BLEU** | NLTK (`sentence_bleu`) | N-gram precision w/ smoothing [0.2656] |

## 🔧 Setup

```bash
pip install sentence-transformers==4.57.6 torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 safetensors==0.4.2 bert-score rouge-score rouge nltk


In class take away:

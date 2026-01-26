#  Lecture Notes: NLP Lab 2 - Tokenization Mastery

## **In this lecture, I learned HOW TO TOKENIZE TEXT** and **WHY** we distinguish:

| **Level** | **Method** | **Example** | **Use Case** |
|-----------|------------|-------------|--------------|
| **Sentence** | `re.split(r'[.?!\n]\s+[A-Z]')` | `"Fulbright..."` | Document splitting |
| **Word** | `re.findall(r'\w+')` | `"Fulbright", "e.g."` | Basic NLP |
| **Character** | `list(text)` | `['F', 'u', 'l'...]` | Rare, spell check |
| **Subword** | GPT-2 BPE | `[37, 377, 29199]` | **LLM MAGIC** |

## **PROF'S 2 REASONS TO LOVE TOKEN LEVEL** 

### **Reason 1: AI Charges by TOKENS (not sentences!)**
Same meaning, different costs:
"Fulbright is great" → 4 tokens → $0.0004
"A beautiful school named Fulbright" → 12 tokens → $0.0012
**OpenAI bills: $2.50/million tokens** - shorter tokens = $$$ saved!

### **Reason 2: SUBWORD = SPACE & TIME SAVINGS** 

Without subword (word-level):

"organization", "organizations", "organizational" → 3 vocab entries 

Vocab size explodes → OOM errors

With BPE subword:

"organ" + "ization" → 1 shared prefix 

"organ" + "ization" + "al" → reuse "organ", "ization"

Vocab stays small → trains 10x faster

Tokenization isn't preprocessing - it's COMPRESSION ENGINEERING"

Sentences → too slow for billing

Words → vocab explosion

Characters → too sparse

SUBWORDS = Sweet spot
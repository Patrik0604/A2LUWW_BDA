# This is a document similarity evaluation service.
- It needs 2 input documents to read: reference.txt and generated.txt
- There is 2 output:
- If you run in the CLI the main.py it returns a report.
- If you call the API with the 2 documents it returns the 6 key metric.
- Test data attached
- The first run may take a 10 minutes, because it needs to download the models and libs (~2-3GB) but after that the run takes about 1-2sec

## Features
- Semantic similarity (Sentence-BERT + cosine similarity)
- Lexical overlap (ROUGE-L)
- Contextual token similarity (BERTScore)
- Numeric consistency check (numeric extraction + Jaccard similarity)
- Composite textual and overall similarity indices
- FastAPI-based REST interface

### API
- POST /compare-files with two .txt files (reference, generated).
#### Response JSON:
- cosine_similarity
- rougeL_f1
- bertscore_f1
- numeric_jaccard
- textual_index
- overall_index

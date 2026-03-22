# 1. Set up the environment 

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv 
source .venv/bin/activate  
pip install -r requirements.txt
```


# 2. Prepare knowledge base

Run the following script to process the raw corpus and benchmark files:

```bash
python src/prepare_data.py \             
  --corpus_dir data/raw/corpus \               
  --benchmarks_dir data/raw/benchmarks \                  
  --output_dir data/processed
```

This step will:
- read the raw legal corpus
- read the benchmark datasets
- convert them into processed JSON files
- save outputs under data/processed


# 3. Build retrieval indexes

Run the following to create both fixed-chunk and recursive-chunk indexes:
```bash
python src/build_indexes.py \
  --corpus_path data/processed/corpus.json \
  --output_dir artifacts
```

This step will:
- read processed corpus
- do fixed chunking and recursive chunking respectively
- save chunks into JSONL
- build two FAISS index: fixed chunks and recursive chunks
- save index and metadata
- output a summary for checking

After running this step, the outputs will be stored under artifacts/


# 4. Prepare the mini evaluation set

To generate the LegalBench-RAG-mini question set, run:

```bash
python src/legalbench_rag_mini.py
```

This script selects exactly 194 queries from each of the four datasets, as suggested from the LegalBench-RAG paper.

The output question file will be:
```bash
data/processed/questions_mini.json
```


# 5. Run the experiments

## To test the pipeline without LLM generator
```bash
python src/run_experiments_hf_local_batched.py \
  --questions_path data/processed/questions_mini.json \  
  --llm_provider dummy \
  --output_dir outputs_dummy \
  --question_limit 10
```
---

## To run the experiments with LLM generator

### Option1: Run on a local machine using Ollama
Step 1. Install Ollama for macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Step 2. Download the model
```bash
ollama pull qwen3:8b
```

Step 3. Start the Ollama server
```bash
ollama serve
```

Step 4. Run the experiment
```bash
python src/run_experiments_hf_local_batched.py \
  --questions_path data/processed/questions_mini.json \  
  --llm_provider ollama \
  --generator_model qwen3:8b \
  --ollama_base_url http://localhost:11434/v1 \
  --output_dir outputs_llm_only \
  --start_index 0 \
  --end_index 100 \
  --run_names llm_only
```

### Option2: Run on a GPU server using a Hugging Face local model

For example:
```bash
python src/run_experiments_hf_local_batched.py \
  --questions_path data/processed/questions_mini.json \
  --llm_provider hf_local \
  --generator_model Qwen/Qwen3-8B \
  --hf_device_map auto \
  --hf_torch_dtype float16 \
  --hf_max_new_tokens 256 \
  --hf_batch_size 4 \
  --batch_size 4 \
  --output_dir outputs_llm_only \
  --start_index 0 \
  --end_index 100 \
  --run_names llm_only \
  --suppress_hf_warnings
```

Parameters:
- `--start_index  --end_index` runs part of the question set
  - `--start_index 0 --end_index 100` run questions 1 to 100
  - `--start_index 100 --end_index 200` run questions 101 to 200
  - if run all the questions: no need to include this
- `--run_names` specifies which pipelines to run
  - `llm_only`: No retrieval, only direct generation from the LLM
  - `standard_rag_fixed`: Standard RAG using the fixed-chunk FAISS index
  - `standard_rag_recursive`: Standard RAG using the recursive-chunk FAISS index
  - `agentic_rag_fixed`: Agentic RAG using the fixed-chunk FAISS index
  - `agentic_rag_recursive`: Agentic RAG using the recursive-chunk FAISS index
 

# 6. Evaluation
## 6.1 Retrieval performance
| Pipeline | Chunking Strategy | Precision@5 | Recall@5 |
|---|---|---|---|
| Standard RAG | Fixed-size | XX.XX | XX.XX |
| Standard RAG | Recursive | XX.XX | XX.XX |
| Agentic RAG | Fixed-size | XX.XX | XX.XX |
| Agentic RAG | Recursive | XX.XX | XX.XX |

## 6.2 Answer performance
| Pipeline | Chunking Strategy | Answer Correctness | Evidence Grounding | Hallucination Rate |
|---|---|---|---|---|
| LLM-only | — | XX.XX | XX.XX | XX.XX |
| Standard RAG | Fixed-size | XX.XX | XX.XX | XX.XX |
| Standard RAG | Recursive | XX.XX | XX.XX | XX.XX |
| Agentic RAG | Fixed-size | XX.XX | XX.XX | XX.XX |
| Agentic RAG | Recursive | XX.XX | XX.XX | XX.XX |

## 6.3 Effect of Agentic Retrieval
| Metric | Standard RAG | Agentic RAG |
|---|---|---|
| Precision@5 | XX.XX | XX.XX |
| Recall@5 | XX.XX | XX.XX |
| Answer Correctness | XX.XX | XX.XX |
| Evidence Grounding | XX.XX | XX.XX |
| Hallucination Rate | XX.XX | XX.XX |

## 6.4 Effect of Chunking Strategy
| Pipeline | Fixed-size Correctness | Recursive Correctness | Fixed-size Grounding | Recursive Grounding |
|---|---|---|---|---|
| Standard RAG | XX.XX | XX.XX | XX.XX | XX.XX |
| Agentic RAG | XX.XX | XX.XX | XX.XX | XX.XX |

---

# Experiment setup
Goal: evaluate whether agentic retrieval strategies can improve legal question answering compared with a conventional standard RAG pipeline

1. Three systems: LLM-only baseline, Standard RAG baseline, Agentic RAG
2. Dataset and knowledge base: LegalBench-RAG-mini as the dataset and LegalBench-RAG corpus as the knowledge base
3. Document chunking strategies:
- Fixed-size chunking: documents are split into chunks of 500 characters with 100 characters of overlap.
- Recursive chunking: documents are split using a recursive text splitter, with an approximate chunk size of 500 characters and 100 characters of overlap, while attempting to preserve sentence or paragraph boundaries.
4. Retrieval Settings:
- embedding model: bge-large-en-v1.5
- store chunk embeddings in FAISS vector index
- retrieve the top 5 chunks for each query
5. Agentic retrieval workflow:
- first retrieves the top-5 chunks
- llm evaluates whether the retrieved evidence is sufficient to answer the question
- if insufficient, the model rewrites the query
- perform second-round top-5 retrieval using the rewritten query
- evidence from both rounds is merged and deduplicated
- maximum total evidence chunks used for final generation: 8
6. LLM generator model and prompting
- model: qwen3:8b 
- use different prompts for three systems
7. Evaluation: retrieval performance/answer performance

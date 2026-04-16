# Instructions of Legal RAG project

## 1. Set up the environment 

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv 
source .venv/bin/activate  
pip install -r requirements.txt
```


## 2. Prepare knowledge base

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


## 3. Build retrieval indexes

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


## 4. Prepare the mini evaluation set

To generate the LegalBench-RAG-mini question set, run:

```bash
python src/legalbench_rag_mini.py
```

This script selects exactly 194 queries from each of the four datasets, as suggested from the LegalBench-RAG paper.

The output question file will be:
```bash
data/processed/questions_mini.json
```


## 5. Run the experiments

### To test the pipeline without LLM generator
```bash
python src/run_experiments_hf_local_batched.py \
  --questions_path data/processed/questions_mini.json \  
  --llm_provider dummy \
  --output_dir outputs_dummy \
  --question_limit 10
```
---

### To run the experiments with LLM generator
For example: Run on a local machine using Ollama

Step 1. Install Ollama for macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Step 2. Download the model
```bash
ollama pull gemma3:12b
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
  --generator_model gemma3:12b \
  --ollama_base_url http://localhost:11434/v1 \
  --output_dir outputs_llm_only \
  --start_index 0 \
  --end_index 100 \
  --run_names llm_only
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


 All the results of experiments are saved in the `outputs` folder
 - gemma_4b
 - qwen_8b


## 6. Evaluation of the outputs
 

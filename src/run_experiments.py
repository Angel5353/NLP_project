from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from indexing import FaissIndexer
from pipelines import (
    LLMOnlyPipeline,
    StandardRAGPipeline,
    AgenticRAGPipeline,
)
from llm_client import build_llm_client


# =========================
# I/O helpers
# =========================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Data loading
# =========================

def load_questions(question_path: str) -> List[Dict[str, Any]]:
    """
    Supports either:
        - a JSON list of question items
        - a JSONL file with one question item per line

    Expected question format:
        {
            "question_id": "...",
            "question": "..."
        }
    """
    if question_path.endswith(".jsonl"):
        questions = load_jsonl(question_path)
    else:
        questions = load_json(question_path)

    if not isinstance(questions, list):
        raise ValueError("Questions file must contain a list of question objects.")

    for q in questions:
        if "question_id" not in q or "question" not in q:
            raise ValueError(
                "Each question item must contain 'question_id' and 'question'."
            )

    return questions


# =========================
# Retriever loading
# =========================

def load_retriever(
    model_name: str,
    index_path: str,
    metadata_path: str,
) -> FaissIndexer:
    retriever = FaissIndexer(model_name=model_name)
    retriever.load(index_path=index_path, metadata_path=metadata_path)
    return retriever


# =========================
# Experiment running
# =========================

def run_pipeline_on_questions(
    pipeline: Any,
    questions: List[Dict[str, Any]],
    run_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run a pipeline over all questions.

    Returns:
        - successful results
        - error records
    """
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for question_item in tqdm(questions, desc=run_name):
        try:
            result = pipeline.run(question_item)
            results.append(result)
        except Exception as e:
            errors.append({
                "run_name": run_name,
                "question_id": question_item.get("question_id"),
                "question": question_item.get("question"),
                "error": str(e),
            })

    return results, errors


def maybe_slice_questions(
    questions: List[Dict[str, Any]],
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    if limit is None or limit <= 0:
        return questions
    return questions[:limit]


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Legal RAG experiments.")

    # data
    parser.add_argument(
        "--questions_path",
        type=str,
        required=True,
        help="Path to questions JSON or JSONL file.",
    )

    # embeddings / retrievers
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model used for FAISS retrieval.",
    )
    parser.add_argument(
        "--fixed_index_path",
        type=str,
        required=True,
        help="Path to fixed chunk FAISS index file.",
    )
    parser.add_argument(
        "--fixed_metadata_path",
        type=str,
        required=True,
        help="Path to fixed chunk metadata pickle file.",
    )
    parser.add_argument(
        "--recursive_index_path",
        type=str,
        required=True,
        help="Path to recursive chunk FAISS index file.",
    )
    parser.add_argument(
        "--recursive_metadata_path",
        type=str,
        required=True,
        help="Path to recursive chunk metadata pickle file.",
    )

    # llm
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="openai",
        choices=["openai", "dummy"],
        help="LLM provider.",
    )
    parser.add_argument(
        "--generator_model",
        type=str,
        default="gpt-4o-mini",
        help="Generator model name.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="Optional explicit OpenAI API key. If omitted, uses OPENAI_API_KEY env var.",
    )

    # experiment controls
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of retrieved chunks per round.",
    )
    parser.add_argument(
        "--max_final_evidence",
        type=int,
        default=8,
        help="Maximum number of evidence chunks used in final agentic generation.",
    )
    parser.add_argument(
        "--question_limit",
        type=int,
        default=0,
        help="Optional debug limit; 0 means run all questions.",
    )

    # output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save experiment results.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) load questions
    questions = load_questions(args.questions_path)
    questions = maybe_slice_questions(
        questions,
        None if args.question_limit == 0 else args.question_limit,
    )

    print(f"Loaded {len(questions)} questions.")

    # 2) load retrievers
    print("Loading fixed retriever...")
    fixed_retriever = load_retriever(
        model_name=args.embedding_model,
        index_path=args.fixed_index_path,
        metadata_path=args.fixed_metadata_path,
    )

    print("Loading recursive retriever...")
    recursive_retriever = load_retriever(
        model_name=args.embedding_model,
        index_path=args.recursive_index_path,
        metadata_path=args.recursive_metadata_path,
    )

    # 3) build llm client
    llm = build_llm_client(
        provider=args.llm_provider,
        model_name=args.generator_model,
        api_key=args.openai_api_key,
    )

    # 4) build pipelines
    pipelines = {
        "llm_only": LLMOnlyPipeline(llm=llm),
        "standard_rag_fixed": StandardRAGPipeline(
            retriever=fixed_retriever,
            llm=llm,
            top_k=args.top_k,
        ),
        "standard_rag_recursive": StandardRAGPipeline(
            retriever=recursive_retriever,
            llm=llm,
            top_k=args.top_k,
        ),
        "agentic_rag_fixed": AgenticRAGPipeline(
            retriever=fixed_retriever,
            llm=llm,
            top_k=args.top_k,
            max_final_evidence=args.max_final_evidence,
        ),
        "agentic_rag_recursive": AgenticRAGPipeline(
            retriever=recursive_retriever,
            llm=llm,
            top_k=args.top_k,
            max_final_evidence=args.max_final_evidence,
        ),
    }

    # 5) run experiments
    all_error_records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for run_name, pipeline in pipelines.items():
        print(f"\nRunning: {run_name}")
        results, errors = run_pipeline_on_questions(
            pipeline=pipeline,
            questions=questions,
            run_name=run_name,
        )

        result_path = output_dir / f"{run_name}.jsonl"
        error_path = output_dir / f"{run_name}_errors.json"
        save_jsonl(str(result_path), results)
        save_json(str(error_path), errors)

        all_error_records.extend(errors)

        summary[run_name] = {
            "num_questions": len(questions),
            "num_successful": len(results),
            "num_errors": len(errors),
            "result_path": str(result_path),
            "error_path": str(error_path),
        }

        print(
            f"Finished {run_name}: "
            f"{len(results)} success, {len(errors)} errors."
        )

    # 6) save global summary
    summary_path = output_dir / "experiment_summary.json"
    save_json(str(summary_path), summary)

    all_errors_path = output_dir / "all_errors.json"
    save_json(str(all_errors_path), all_error_records)

    print("\nAll experiments complete.")
    print(f"Summary saved to: {summary_path}")
    print(f"All errors saved to: {all_errors_path}")


if __name__ == "__main__":
    main()
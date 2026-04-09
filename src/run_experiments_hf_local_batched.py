from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

from indexing import FaissIndexer
from pipelines_batched import (
    LLMOnlyPipeline,
    StandardRAGPipeline,
    AgenticRAGPipeline,
    parse_sufficiency_label,
)
from llm_client_hf_local_batched import build_llm_client


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


def append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_json_if_exists(path: str) -> Any:
    input_path = Path(path)
    if not input_path.exists():
        return None
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_questions(question_path: str) -> List[Dict[str, Any]]:
    questions = load_jsonl(question_path) if question_path.endswith(".jsonl") else load_json(question_path)
    if not isinstance(questions, list):
        raise ValueError("Questions file must contain a list of question objects.")
    for q in questions:
        if "question_id" not in q or "question" not in q:
            raise ValueError("Each question item must contain 'question_id' and 'question'.")
    return questions


def load_retriever(model_name: str, index_path: str, metadata_path: str) -> FaissIndexer:
    retriever = FaissIndexer(model_name=model_name)
    retriever.load(index_path=index_path, metadata_path=metadata_path)
    return retriever


def maybe_slice_questions(
    questions: List[Dict[str, Any]],
    start_index: int = 0,
    end_index: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    sliced = questions[start_index:end_index]
    if limit is None or limit <= 0:
        return sliced
    return sliced[:limit]


def parse_selected_run_names(run_names_arg: str) -> List[str]:
    default_run_names = [
        "llm_only",
        "standard_rag_fixed",
        "standard_rag_recursive",
        "agentic_rag_fixed",
        "agentic_rag_recursive",
    ]
    if not run_names_arg.strip():
        return default_run_names
    return [name.strip() for name in run_names_arg.split(",") if name.strip()]


def ensure_file_exists(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def batched(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def run_llm_only_batched(
    pipeline: LLMOnlyPipeline,
    questions: List[Dict[str, Any]],
    run_name: str,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for batch in tqdm(batched(questions, batch_size), desc=run_name):
        prompts = [pipeline.build_prompt(q) for q in batch]
        try:
            answers = pipeline.llm.generate_batch(prompts, temperature=0.0)
            for question_item, answer in zip(batch, answers):
                results.append(pipeline.make_result(question_item, answer))
        except Exception as e:
            for question_item in batch:
                errors.append({
                    "run_name": run_name,
                    "question_id": question_item.get("question_id"),
                    "question": question_item.get("question"),
                    "error": str(e),
                })
    return results, errors


def run_standard_rag_batched(
    pipeline: StandardRAGPipeline,
    questions: List[Dict[str, Any]],
    run_name: str,
    batch_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    states: List[Dict[str, Any]] = []
    for question_item in tqdm(questions, desc=f"{run_name}:retrieve"):
        try:
            retrieved_chunks = pipeline.retrieve(question_item["question"])
            prompt = pipeline.build_prompt(question_item["question"], retrieved_chunks)
            states.append({
                "question_item": question_item,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
            })
        except Exception as e:
            errors.append({
                "run_name": run_name,
                "question_id": question_item.get("question_id"),
                "question": question_item.get("question"),
                "error": str(e),
            })

    for batch_states in tqdm(batched(states, batch_size), desc=f"{run_name}:generate"):
        prompts = [s["prompt"] for s in batch_states]
        try:
            answers = pipeline.llm.generate_batch(prompts, temperature=0.0)
            for state, answer in zip(batch_states, answers):
                results.append(
                    pipeline.make_result(
                        state["question_item"],
                        state["retrieved_chunks"],
                        answer,
                    )
                )
        except Exception as e:
            for state in batch_states:
                q = state["question_item"]
                errors.append({
                    "run_name": run_name,
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                    "error": str(e),
                })

    return results, errors


def run_agentic_rag_batched(
    pipeline: AgenticRAGPipeline,
    questions: List[Dict[str, Any]],
    run_name: str,
    batch_size: int,
    second_round_top_k_multiplier: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    states: List[Dict[str, Any]] = []
    for question_item in tqdm(questions, desc=f"{run_name}:retrieve1"):
        try:
            first_round = pipeline.retrieve(question_item["question"])
            states.append({
                "question_item": question_item,
                "first_round": first_round,
                "sufficiency_judgment": None,
                "rewritten_query": None,
                "second_round": [],
                "final_evidence": [],
            })
        except Exception as e:
            errors.append({
                "run_name": run_name,
                "question_id": question_item.get("question_id"),
                "question": question_item.get("question"),
                "error": str(e),
            })

    for batch_states in tqdm(batched(states, batch_size), desc=f"{run_name}:judge"):
        prompts = [
            pipeline.build_sufficiency_prompt(s["question_item"]["question"], s["first_round"])
            for s in batch_states
        ]
        try:
            raw_labels = pipeline.llm.generate_batch(prompts, temperature=0.0)
            for state, raw_label in zip(batch_states, raw_labels):
                state["sufficiency_judgment"] = parse_sufficiency_label(raw_label)
        except Exception as e:
            for state in batch_states:
                q = state["question_item"]
                errors.append({
                    "run_name": run_name,
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                    "error": str(e),
                })

    rewrite_candidates = [s for s in states if s.get("sufficiency_judgment") == "insufficient"]
    for batch_states in tqdm(batched(rewrite_candidates, batch_size), desc=f"{run_name}:rewrite"):
        prompts = [
            pipeline.build_rewrite_prompt(s["question_item"]["question"], s["first_round"])
            for s in batch_states
        ]
        try:
            rewritten_queries = pipeline.llm.generate_batch(prompts, temperature=0.0)
            for state, rewritten_query in zip(batch_states, rewritten_queries):
                state["rewritten_query"] = rewritten_query.strip()
        except Exception as e:
            for state in batch_states:
                q = state["question_item"]
                errors.append({
                    "run_name": run_name,
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                    "error": str(e),
                })

    for state in tqdm(rewrite_candidates, desc=f"{run_name}:retrieve2"):
        if not state.get("rewritten_query"):
            continue
        try:
            state["second_round"] = pipeline.retrieve(
                state["rewritten_query"],
                top_k=pipeline.top_k * max(1, second_round_top_k_multiplier),
            )
        except Exception as e:
            q = state["question_item"]
            errors.append({
                "run_name": run_name,
                "question_id": q.get("question_id"),
                "question": q.get("question"),
                "error": str(e),
            })

    valid_states: List[Dict[str, Any]] = []
    final_prompts: List[str] = []
    for state in states:
        try:
            state["final_evidence"] = pipeline.prepare_final_evidence(state["first_round"], state["second_round"])
            final_prompts.append(
                pipeline.build_final_prompt(
                    state["question_item"]["question"],
                    state["final_evidence"],
                )
            )
            valid_states.append(state)
        except Exception as e:
            q = state["question_item"]
            errors.append({
                "run_name": run_name,
                "question_id": q.get("question_id"),
                "question": q.get("question"),
                "error": str(e),
            })

    for idx, batch_states in enumerate(tqdm(batched(valid_states, batch_size), desc=f"{run_name}:final")):
        prompt_batch = final_prompts[idx * batch_size:(idx + 1) * batch_size]
        try:
            answers = pipeline.llm.generate_batch(prompt_batch, temperature=0.0)
            for state, answer in zip(batch_states, answers):
                results.append(
                    pipeline.make_result(
                        question_item=state["question_item"],
                        first_round=state["first_round"],
                        sufficiency_judgment=state.get("sufficiency_judgment") or "insufficient",
                        rewritten_query=state.get("rewritten_query"),
                        second_round=state.get("second_round", []),
                        final_evidence=state.get("final_evidence", []),
                        final_answer=answer,
                    )
                )
        except Exception as e:
            for state in batch_states:
                q = state["question_item"]
                errors.append({
                    "run_name": run_name,
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                    "error": str(e),
                })

    return results, errors


def run_pipeline_on_questions(
    pipeline: Any,
    questions: List[Dict[str, Any]],
    run_name: str,
    batch_size: int,
    second_round_top_k_multiplier: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if isinstance(pipeline, LLMOnlyPipeline):
        return run_llm_only_batched(pipeline, questions, run_name, batch_size)
    if isinstance(pipeline, StandardRAGPipeline):
        return run_standard_rag_batched(pipeline, questions, run_name, batch_size)
    if isinstance(pipeline, AgenticRAGPipeline):
        return run_agentic_rag_batched(
            pipeline,
            questions,
            run_name,
            batch_size,
            second_round_top_k_multiplier,
        )

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for question_item in tqdm(questions, desc=run_name):
        try:
            results.append(pipeline.run(question_item))
        except Exception as e:
            errors.append({
                "run_name": run_name,
                "question_id": question_item.get("question_id"),
                "question": question_item.get("question"),
                "error": str(e),
            })
    return results, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Legal RAG experiments with per-question retrieval and batched generation.")
    parser.add_argument("--questions_path", type=str, default="data/processed/questions.json")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--fixed_index_path", type=str, default="artifacts/indexes/fixed_index.faiss")
    parser.add_argument("--fixed_metadata_path", type=str, default="artifacts/indexes/fixed_metadata.pkl")
    parser.add_argument("--recursive_index_path", type=str, default="artifacts/indexes/recursive_index.faiss")
    parser.add_argument("--recursive_metadata_path", type=str, default="artifacts/indexes/recursive_metadata.pkl")
    parser.add_argument("--llm_provider", type=str, default="openai", choices=["openai", "ollama", "hf_local", "dummy"])
    parser.add_argument("--generator_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434/v1")
    parser.add_argument("--hf_max_new_tokens", type=int, default=256)
    parser.add_argument("--hf_device_map", type=str, default="auto")
    parser.add_argument("--hf_torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--hf_trust_remote_code", action="store_true")
    parser.add_argument("--hf_batch_size", type=int, default=4, help="HF pipeline batch size for generation.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_final_evidence", type=int, default=8)
    parser.add_argument("--question_limit", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    parser.add_argument("--run_names", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for grouped generation calls.")
    parser.add_argument("--agentic_second_round_top_k_multiplier", type=int, default=1)
    parser.add_argument("--suppress_hf_warnings", action="store_true")
    args = parser.parse_args()

    if args.suppress_hf_warnings:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_questions = load_questions(args.questions_path)
    end_index = None if args.end_index == -1 else args.end_index
    questions = maybe_slice_questions(
        all_questions,
        start_index=args.start_index,
        end_index=end_index,
        limit=None if args.question_limit == 0 else args.question_limit,
    )

    selected_end_display = args.start_index + len(questions)
    print(f"Loaded {len(all_questions)} total questions.")
    print(f"Running questions in slice [{args.start_index}:{selected_end_display}] ({len(questions)} questions selected).")

    available_run_names = [
        "llm_only",
        "standard_rag_fixed",
        "standard_rag_recursive",
        "agentic_rag_fixed",
        "agentic_rag_recursive",
    ]
    selected_run_names = parse_selected_run_names(args.run_names)
    invalid_run_names = [name for name in selected_run_names if name not in available_run_names]
    if invalid_run_names:
        raise ValueError(f"Unknown run_names: {invalid_run_names}. Available options are: {available_run_names}")
    print(f"Selected pipelines: {selected_run_names}")

    need_fixed_retriever = any(name in {"standard_rag_fixed", "agentic_rag_fixed"} for name in selected_run_names)
    need_recursive_retriever = any(name in {"standard_rag_recursive", "agentic_rag_recursive"} for name in selected_run_names)

    fixed_retriever = None
    recursive_retriever = None
    if need_fixed_retriever:
        ensure_file_exists(args.fixed_index_path, "fixed index")
        ensure_file_exists(args.fixed_metadata_path, "fixed metadata")
        print("Loading fixed retriever...")
        fixed_retriever = load_retriever(args.embedding_model, args.fixed_index_path, args.fixed_metadata_path)
    if need_recursive_retriever:
        ensure_file_exists(args.recursive_index_path, "recursive index")
        ensure_file_exists(args.recursive_metadata_path, "recursive metadata")
        print("Loading recursive retriever...")
        recursive_retriever = load_retriever(args.embedding_model, args.recursive_index_path, args.recursive_metadata_path)

    llm = build_llm_client(
        provider=args.llm_provider,
        model_name=args.generator_model,
        api_key=(args.openai_api_key if args.llm_provider == "openai" else None),
        base_url=(args.ollama_base_url if args.llm_provider == "ollama" else None),
        hf_max_new_tokens=args.hf_max_new_tokens,
        hf_device_map=args.hf_device_map,
        hf_torch_dtype=args.hf_torch_dtype,
        hf_trust_remote_code=args.hf_trust_remote_code,
        hf_batch_size=args.hf_batch_size,
    )

    pipelines: Dict[str, Any] = {}
    if "llm_only" in selected_run_names:
        pipelines["llm_only"] = LLMOnlyPipeline(llm=llm)
    if "standard_rag_fixed" in selected_run_names:
        pipelines["standard_rag_fixed"] = StandardRAGPipeline(retriever=fixed_retriever, llm=llm, top_k=args.top_k)
    if "standard_rag_recursive" in selected_run_names:
        pipelines["standard_rag_recursive"] = StandardRAGPipeline(retriever=recursive_retriever, llm=llm, top_k=args.top_k)
    if "agentic_rag_fixed" in selected_run_names:
        pipelines["agentic_rag_fixed"] = AgenticRAGPipeline(retriever=fixed_retriever, llm=llm, top_k=args.top_k, max_final_evidence=args.max_final_evidence)
    if "agentic_rag_recursive" in selected_run_names:
        pipelines["agentic_rag_recursive"] = AgenticRAGPipeline(retriever=recursive_retriever, llm=llm, top_k=args.top_k, max_final_evidence=args.max_final_evidence)

    all_error_records: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    for run_name, pipeline in pipelines.items():
        print(f"\nRunning: {run_name}")
        results, errors = run_pipeline_on_questions(
            pipeline=pipeline,
            questions=questions,
            run_name=run_name,
            batch_size=args.batch_size,
            second_round_top_k_multiplier=args.agentic_second_round_top_k_multiplier,
        )

        result_path = output_dir / f"{run_name}.jsonl"
        error_path = output_dir / f"{run_name}_errors.json"

        if args.start_index > 0 and result_path.exists():
            append_jsonl(str(result_path), results)
        else:
            save_jsonl(str(result_path), results)

        if args.start_index > 0 and error_path.exists():
            existing_errors = load_existing_json_if_exists(str(error_path)) or []
            if not isinstance(existing_errors, list):
                existing_errors = []
            save_json(str(error_path), existing_errors + errors)
        else:
            save_json(str(error_path), errors)

        all_error_records.extend(errors)
        summary[run_name] = {
            "slice_start_index": args.start_index,
            "slice_end_index_exclusive": selected_end_display,
            "num_questions": len(questions),
            "num_successful": len(results),
            "num_errors": len(errors),
            "result_path": str(result_path),
            "error_path": str(error_path),
        }
        print(f"Finished {run_name}: {len(results)} success, {len(errors)} errors.")

    summary_path = output_dir / "experiment_summary.json"
    save_json(str(summary_path), summary)

    all_errors_path = output_dir / "all_errors.json"
    if args.start_index > 0 and all_errors_path.exists():
        existing_all_errors = load_existing_json_if_exists(str(all_errors_path)) or []
        if not isinstance(existing_all_errors, list):
            existing_all_errors = []
        save_json(str(all_errors_path), existing_all_errors + all_error_records)
    else:
        save_json(str(all_errors_path), all_error_records)

    print("\nAll experiments complete.")
    print(f"Summary saved to: {summary_path}")
    print(f"All errors saved to: {all_errors_path}")


if __name__ == "__main__":
    main()

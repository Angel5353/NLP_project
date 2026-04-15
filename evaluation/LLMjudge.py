import os
import json
import time
import argparse
from collections import Counter
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm
import os
from getpass import getpass
from pathlib import Path

OUTPUTS_ROOT = Path("../outputs")
RUN_FILES = {
    "gemma_4b_llm_only": OUTPUTS_ROOT / "gemma_4b" / "llm_only.jsonl",
    "gemma_4b_standard_rag_fixed": OUTPUTS_ROOT / "gemma_4b" / "standard_rag_fixed.jsonl",
    "gemma_4b_standard_rag_recursive": OUTPUTS_ROOT / "gemma_4b" / "standard_rag_recursive.jsonl",
    "gemma_4b_agentic_rag_fixed": OUTPUTS_ROOT / "gemma_4b" / "agentic_rag_fixed.jsonl",
    "gemma_4b_agentic_rag_recursive": OUTPUTS_ROOT / "gemma_4b" / "agentic_rag_recursive.jsonl",
    "qwen_8b_llm_only": OUTPUTS_ROOT / "qwen_8b" / "llm_only.jsonl",
    "qwen_8b_standard_rag_fixed": OUTPUTS_ROOT / "qwen_8b" / "standard_rag_fixed.jsonl",
    "qwen_8b_standard_rag_recursive": OUTPUTS_ROOT / "qwen_8b" / "standard_rag_recursive.jsonl",
    "qwen_8b_agentic_rag_fixed": OUTPUTS_ROOT / "qwen_8b" / "agentic_rag_fixed.jsonl",
    "qwen_8b_agentic_rag_recursive": OUTPUTS_ROOT / "qwen_8b" / "agentic_rag_recursive.jsonl",
}


def safe_get(d: Dict[str, Any], keys: List[str], default=""):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def build_client(hf_token: str) -> OpenAI:
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return {
        "label": "parse_error",
        "score": 0.0,
        "reason": text[:1000]
    }


def chat_json(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    sleep_sec: float = 2.0,
) -> Dict[str, Any]:
    last_err = None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return parse_json_response(content)

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(sleep_sec)

    return {
        "label": "api_error",
        "score": 0.0,
        "reason": last_err or "Unknown API error"
    }


def judge_answer_quality(
    client: OpenAI,
    model_name: str,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    system_prompt = (
        "You are a strict evaluation judge.\n"
        "Evaluate how well the candidate answer addresses the user's question.\n"
        "Return JSON only with keys: label, score, reason.\n"
        "label must be exactly one of: fully_relevant, partially_relevant, irrelevant.\n"
        "score must be a float between 0 and 1.\n"
        "Judge based on relevance, completeness, and clarity.\n"
        "Do not require outside factual verification.\n"
        "Do not include any extra text outside JSON."
    )

    user_prompt = f"""
Question:
{question}

Candidate Answer:
{answer}

Instructions:
- "fully_relevant" means the answer directly addresses the question and is reasonably complete.
- "partially_relevant" means the answer is somewhat relevant but incomplete, vague, or only partially addresses the question.
- "irrelevant" means it does not answer the question or is mostly off-topic.
- Do not judge based on hidden world knowledge.
- Focus on whether the answer responds appropriately to the question.

Return JSON only.
""".strip()

    result = chat_json(client, model_name, system_prompt, user_prompt)

    label = str(result.get("label", "irrelevant")).strip().lower()
    if label not in {"fully_relevant", "partially_relevant", "irrelevant"}:
        label = "irrelevant"

    try:
        score = float(result.get("score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))

    return {
        "label": label,
        "score": score,
        "reason": str(result.get("reason", "")),
    }


def judge_groundedness(
    client: OpenAI,
    model_name: str,
    question: str,
    answer: str,
    evidence: str,
) -> Dict[str, Any]:
    system_prompt = (
        "You are a strict RAG evaluation judge.\n"
        "Evaluate whether the candidate answer is grounded in the provided evidence only.\n"
        "Do not use outside knowledge.\n"
        "Return JSON only with keys: label, score, reason.\n"
        "label must be exactly one of: grounded, partially_grounded, ungrounded.\n"
        "score must be a float between 0 and 1.\n"
        "If the answer contains unsupported claims, penalize it.\n"
        "Do not include any extra text outside JSON."
    )

    user_prompt = f"""
Question:
{question}

Candidate Answer:
{answer}

Evidence:
{evidence}

Instructions:
- "grounded" means all important claims in the answer are supported by the evidence.
- "partially_grounded" means the answer is partly supported, but includes some unsupported or weakly supported claims.
- "ungrounded" means the key claims are not supported by the evidence, or the answer hallucinates.
- Use only the evidence above. Ignore outside world knowledge.

Return JSON only.
""".strip()

    result = chat_json(client, model_name, system_prompt, user_prompt)

    label = str(result.get("label", "ungrounded")).strip().lower()
    if label not in {"grounded", "partially_grounded", "ungrounded"}:
        label = "ungrounded"

    try:
        score = float(result.get("score", 0.0))
    except Exception:
        score = 0.0
    score = max(0.0, min(1.0, score))

    return {
        "label": label,
        "score": score,
        "reason": str(result.get("reason", "")),
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "num_records": 0,
            "avg_answer_quality_score": 0.0,
            "avg_groundedness_score": 0.0,
            "answer_quality_labels": {},
            "groundedness_labels": {},
        }

    quality_scores = [x["answer_quality"]["score"] for x in results]
    grounded_scores = [x["groundedness"]["score"] for x in results]

    quality_labels = [x["answer_quality"]["label"] for x in results]
    grounded_labels = [x["groundedness"]["label"] for x in results]

    combo_counter = Counter(
        (x["answer_quality"]["label"], x["groundedness"]["label"])
        for x in results
    )

    return {
        "num_records": n,
        "avg_answer_quality_score": sum(quality_scores) / n,
        "avg_groundedness_score": sum(grounded_scores) / n,
        "answer_quality_labels": dict(Counter(quality_labels)),
        "groundedness_labels": dict(Counter(grounded_labels)),
        "joint_label_counts": {
            f"{k1}__{k2}": v for (k1, k2), v in combo_counter.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    parser.add_argument("--output_dir", type=str, default="./results_llm_judge")
    args = parser.parse_args()

    HF_TOKEN = getpass("INPUT YOUR HF TOKEN HERE: ")
    os.environ["HF_TOKEN"] = HF_TOKEN
    client = build_client(HF_TOKEN)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for run_name, input_path in RUN_FILES.items():
        input_path = Path(input_path)

        if not input_path.exists():
            print(f"[SKIP] Missing file: {input_path}")
            continue

        print(f"\nEvaluating {run_name} ...")

        rows = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if args.limit is not None:
            rows = rows[:args.limit]

        results = []

        for row in tqdm(rows, desc=f"LLM judging {run_name}"):
            question_id = safe_get(row, ["question_id", "id"], default=None)
            question = safe_get(row, ["question", "original_question", "query"])
            answer = safe_get(
                row,
                ["answer", "generated_answer", "prediction", "response", "final_answer"]
            )

            retrieved_chunks = row.get("retrieved_chunks", [])
            if isinstance(retrieved_chunks, list) and retrieved_chunks:
                evidence = "\n\n".join(
                    chunk.get("text", "") for chunk in retrieved_chunks if isinstance(chunk, dict)
                )
            else:
                final_evidence = row.get("final_evidence", [])
                if isinstance(final_evidence, list) and final_evidence:
                    evidence = "\n\n".join(
                        chunk.get("text", "") for chunk in final_evidence if isinstance(chunk, dict)
                    )
                else:
                    evidence = safe_get(
                        row,
                        ["evidence", "context", "retrieved_context", "retrieved_passages"],
                        default=""
                    )

            if isinstance(evidence, list):
                evidence = "\n\n".join(str(x) for x in evidence)

            answer_quality = judge_answer_quality(
                client=client,
                model_name=args.model,
                question=question,
                answer=answer,
            )

            groundedness = judge_groundedness(
                client=client,
                model_name=args.model,
                question=question,
                answer=answer,
                evidence=evidence,
            )

            out = {
                "system": run_name,
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "evidence": evidence,
                "answer_quality": answer_quality,
                "groundedness": groundedness,
            }
            results.append(out)

        per_question_path = output_dir / f"{run_name}_per_question.jsonl"
        summary_path = output_dir / f"{run_name}_summary.json"

        with open(per_question_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary = summarize(results)
        summary["system"] = run_name
        summary["input_file"] = str(input_path)
        summary["num_input_rows_used"] = len(results)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        all_summaries.append(summary)

        print(f"Saved per-question results to: {per_question_path}")
        print(f"Saved summary to: {summary_path}")

    all_summary_path = output_dir / "all_summary.json"
    with open(all_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print(f"\nSaved aggregated summary to: {all_summary_path}")


if __name__ == "__main__":
    main()
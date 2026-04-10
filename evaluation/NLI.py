import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


#load paths for questions and files
QUESTIONS_PATH = "questions.json"

RUN_FILES = {
    "llm_only": "llm_only.jsonl",
    "standard_rag_fixed": "standard_rag_fixed.jsonl",
    "standard_rag_recursive": "standard_rag_recursive.jsonl",
    "agentic_rag_fixed": "agentic_rag_fixed.jsonl",
    "agentic_rag_recursive": "agentic_rag_recursive.jsonl",
}

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# extract key word from json files

def extract_generated_answer(record: Dict[str, Any]) -> str:
    candidate_fields = [
        "final_answer",
        "answer",
        "generated_answer",
        "response",
        "prediction",
    ]
    for field in candidate_fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_gold_evidence_text(question_item: Dict[str, Any]) -> str:
    evs = question_item.get("gold_evidence", [])
    parts = []

    for ev in evs:
        text = ev.get("text", "")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    return "\n\n".join(parts)


#load NLI model 

def load_nli_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_label_mapping(model) -> Dict[int, str]:
    id2label = model.config.id2label
    mapping = {}
    for k, v in id2label.items():
        label = str(v).lower()
        if "entail" in label:
            mapping[int(k)] = "entailment"
        elif "contrad" in label:
            mapping[int(k)] = "contradiction"
        else:
            mapping[int(k)] = "neutral"
    return mapping


@torch.no_grad()
def run_nli(premise: str, hypothesis: str, tokenizer, model) -> Dict[str, Any]:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()

    label_map = get_label_mapping(model)

    scored = []
    for i, p in enumerate(probs.tolist()):
        scored.append((label_map[i], p))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    top_label, top_score = scored[0]

    prob_dict = {label: score for label, score in scored}

    return {
        "label": top_label,
        "confidence": top_score,
        "entailment_score": prob_dict.get("entailment", 0.0),
        "neutral_score": prob_dict.get("neutral", 0.0),
        "contradiction_score": prob_dict.get("contradiction", 0.0),
    }


# map metrics to scores

def grounding_score_from_nli(label: str) -> float:
    if label == "entailment":
        return 1.0
    if label == "neutral":
        return 0.5
    return 0.0


def hallucination_from_nli(label: str) -> int:
    return 0 if label == "entailment" else 1


# evaluation

def evaluate_run(
    run_name: str,
    run_file: str,
    questions_by_id: Dict[str, Dict[str, Any]],
    tokenizer,
    model,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = load_jsonl(run_file)

    per_question_rows = []

    missing_qids = 0
    missing_answers = 0
    missing_gold_evidence = 0

    total_grounding = 0.0
    total_hallucination = 0

    label_counts = {
        "entailment": 0,
        "neutral": 0,
        "contradiction": 0,
    }

    for rec in records:
        qid = rec.get("question_id")
        if qid not in questions_by_id:
            missing_qids += 1
            continue

        question_item = questions_by_id[qid]
        gold_evidence_text = extract_gold_evidence_text(question_item)
        #answer = extract_generated_answer(rec)
        answer = rec.get("final_answer", "")
        if "retrieved_chunks" in rec:
            retrieved_text = " ".join([c["text"] for c in rec["retrieved_chunks"]])
        elif "final_evidence" in rec: # Agentic format
            retrieved_text = " ".join([c["text"] for c in rec["final_evidence"]])
        else:
            retrieved_text = ""
            
        premise = retrieved_text

        if not answer:
            missing_answers += 1
            continue

        if not gold_evidence_text:
            missing_gold_evidence += 1
            continue

        nli_result = run_nli(
            #premise=gold_evidence_text,
            premise=premise,
            hypothesis=answer,
            tokenizer=tokenizer,
            model=model,
        )

        label = nli_result["label"]
        grounding_score = grounding_score_from_nli(label)
        hallucination_flag = hallucination_from_nli(label)

        label_counts[label] += 1
        total_grounding += grounding_score
        total_hallucination += hallucination_flag

        per_question_rows.append({
            "question_id": qid,
            "system": run_name,
            "nli_label": label,
            "nli_confidence": nli_result["confidence"],
            "entailment_score": nli_result["entailment_score"],
            "neutral_score": nli_result["neutral_score"],
            "contradiction_score": nli_result["contradiction_score"],
            "grounding_score": grounding_score,
            "hallucination": hallucination_flag,
            "answer": answer,
            "gold_evidence_text": gold_evidence_text,
        })

    n = len(per_question_rows)

    if n == 0:
        summary = {
            "system": run_name,
            "num_records": len(records),
            "num_evaluated": 0,
            "missing_question_ids": missing_qids,
            "missing_answers": missing_answers,
            "missing_gold_evidence": missing_gold_evidence,
            "evidence_grounding_score": 0.0,
            "hallucination_rate": 0.0,
            "entailment_rate": 0.0,
            "neutral_rate": 0.0,
            "contradiction_rate": 0.0,
        }
        return per_question_rows, summary

    summary = {
        "system": run_name,
        "num_records": len(records),
        "num_evaluated": n,
        "missing_question_ids": missing_qids,
        "missing_answers": missing_answers,
        "missing_gold_evidence": missing_gold_evidence,
        "evidence_grounding_score": total_grounding / n,
        "hallucination_rate": total_hallucination / n,
        "entailment_rate": label_counts["entailment"] / n,
        "neutral_rate": label_counts["neutral"] / n,
        "contradiction_rate": label_counts["contradiction"] / n,
    }

    return per_question_rows, summary


def main() -> None:
    print(f"Loading NLI model: {MODEL_NAME}")
    print(f"Using device: {DEVICE}")
    tokenizer, model = load_nli_model(MODEL_NAME)

    questions = load_json(QUESTIONS_PATH)
    questions_by_id = {q["question_id"]: q for q in questions}

    all_rows = []
    all_summaries = []

    for run_name, run_file in RUN_FILES.items():
        if not Path(run_file).exists():
            print(f"[SKIP] Missing run file: {run_file}")
            continue

        print(f"Evaluating {run_name} ...")
        rows, summary = evaluate_run(
            run_name=run_name,
            run_file=run_file,
            questions_by_id=questions_by_id,
            tokenizer=tokenizer,
            model=model,
        )
        all_rows.extend(rows)
        all_summaries.append(summary)

    save_jsonl("grounding_nli_per_question.jsonl", all_rows)
    save_json("grounding_nli_summary.json", all_summaries)

    print("\nDone.")
    print("Saved per-question results to grounding_nli_per_question.jsonl")
    print("Saved summary results to grounding_nli_summary.json")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
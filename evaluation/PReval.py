import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple


# =========================
# Paths
# =========================

QUESTIONS_PATH = "questions.json"
FIXED_CHUNKS_PATH = "fixed_chunks.jsonl"
RECURSIVE_CHUNKS_PATH = "recursive_chunks.jsonl"

RUN_FILES = {
    "standard_rag_fixed": "standard_rag_fixed.jsonl",
    "standard_rag_recursive": "standard_rag_recursive.jsonl",
    "agentic_rag_fixed": "agentic_rag_fixed.jsonl",
    "agentic_rag_recursive": "agentic_rag_recursive.jsonl",
}

TOP_K = 5


# =========================
# Basic I/O
# =========================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# =========================
# Utilities
# =========================

def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def build_doc_to_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    doc_to_chunks = defaultdict(list)
    for ch in chunks:
        doc_to_chunks[ch["doc_id"]].append(ch)
    return doc_to_chunks


def map_gold_evidence_to_chunk_ids(
    question_item: Dict[str, Any],
    doc_to_chunks: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    relevant = set()

    for ev in question_item.get("gold_evidence", []):
        doc_id = ev.get("doc_id")
        ev_start = ev.get("start_char")
        ev_end = ev.get("end_char")

        if doc_id is None or ev_start is None or ev_end is None:
            continue

        for ch in doc_to_chunks.get(doc_id, []):
            ch_start = ch.get("start_char")
            ch_end = ch.get("end_char")
            if ch_start is None or ch_end is None:
                continue

            if spans_overlap(ch_start, ch_end, ev_start, ev_end):
                relevant.add(ch["chunk_id"])

    return sorted(relevant)


# =========================
# Metrics
# =========================

def precision_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    retrieved_k = retrieved_ids[:k]
    if k <= 0:
        return 0.0
    hits = len(set(retrieved_k) & set(gold_ids))
    return hits / k


def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int = 5) -> float:
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & gold_set)
    return hits / len(gold_set)


def recall_all_evidence(retrieved_ids: List[str], gold_ids: List[str]) -> float:
    gold_set = set(gold_ids)
    if not gold_set:
        return 0.0
    hits = len(set(retrieved_ids) & gold_set)
    return hits / len(gold_set)


def gold_gain_over_standard(
    standard_ids_topk: List[str],
    agentic_final_ids: List[str],
    gold_ids: List[str],
) -> int:
    std_hits = set(standard_ids_topk) & set(gold_ids)
    ag_hits = set(agentic_final_ids) & set(gold_ids)
    return len(ag_hits - std_hits)


def extra_gold_coverage(
    standard_ids_topk: List[str],
    agentic_final_ids: List[str],
    gold_ids: List[str],
) -> float:
    extra = set(agentic_final_ids) - set(standard_ids_topk)
    if not extra:
        return 0.0
    return len(extra & set(gold_ids)) / len(extra)


# =========================
# Retrieved chunk parsing
# =========================

def extract_chunk_ids_from_list_field(items: Any) -> List[str]:
    out = []
    if not isinstance(items, list):
        return out

    for item in items:
        if isinstance(item, dict) and "chunk_id" in item:
            out.append(str(item["chunk_id"]))
        elif isinstance(item, str):
            out.append(item)

    return out


def dedup_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def extract_chunk_id_list_from_standard_record(record: Dict[str, Any]) -> List[str]:
    if "retrieved_chunk_ids" in record and isinstance(record["retrieved_chunk_ids"], list):
        return [str(x) for x in record["retrieved_chunk_ids"]]

    if "retrieved_chunks" in record and isinstance(record["retrieved_chunks"], list):
        return extract_chunk_ids_from_list_field(record["retrieved_chunks"])

    if "contexts" in record and isinstance(record["contexts"], list):
        return extract_chunk_ids_from_list_field(record["contexts"])

    return []


def extract_agentic_first_round(record: Dict[str, Any]) -> List[str]:
    for key in ["first_round_retrieval", "first_round"]:
        if key in record and isinstance(record[key], list):
            return extract_chunk_ids_from_list_field(record[key])

    # fallback: if no explicit first round field, reuse retrieved_chunk_ids
    if "retrieved_chunk_ids" in record and isinstance(record["retrieved_chunk_ids"], list):
        return [str(x) for x in record["retrieved_chunk_ids"]]

    return []


def extract_agentic_second_round(record: Dict[str, Any]) -> List[str]:
    for key in ["second_round_retrieval", "second_round"]:
        if key in record and isinstance(record[key], list):
            return extract_chunk_ids_from_list_field(record[key])
    return []


def extract_agentic_final_evidence(record: Dict[str, Any]) -> List[str]:
    if "final_evidence" in record and isinstance(record["final_evidence"], list):
        out = extract_chunk_ids_from_list_field(record["final_evidence"])
        if out:
            return out

    # fallback: merge first + second
    merged = extract_agentic_first_round(record) + extract_agentic_second_round(record)
    return dedup_preserve_order(merged)


def extract_retrieved_chunk_ids(run_name: str, record: Dict[str, Any]) -> List[str]:
    if run_name.startswith("agentic_rag"):
        return extract_agentic_final_evidence(record)
    return extract_chunk_id_list_from_standard_record(record)


# =========================
# Evaluation
# =========================

def evaluate_run(
    run_name: str,
    run_file: str,
    questions_by_id: Dict[str, Dict[str, Any]],
    fixed_doc_to_chunks: Dict[str, List[Dict[str, Any]]],
    recursive_doc_to_chunks: Dict[str, List[Dict[str, Any]]],
    standard_records_by_qid: Dict[str, Dict[str, Any]],
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records = load_jsonl(run_file)

    if run_name.endswith("_fixed"):
        doc_to_chunks = fixed_doc_to_chunks
    elif run_name.endswith("_recursive"):
        doc_to_chunks = recursive_doc_to_chunks
    else:
        raise ValueError(f"Cannot determine chunking strategy from run name: {run_name}")

    per_question_rows = []
    missing_qids = 0
    missing_retrieval = 0

    for rec in records:
        qid = rec.get("question_id")
        if qid not in questions_by_id:
            missing_qids += 1
            continue

        question_item = questions_by_id[qid]
        gold_chunk_ids = map_gold_evidence_to_chunk_ids(question_item, doc_to_chunks)

        retrieved_chunk_ids = extract_retrieved_chunk_ids(run_name, rec)
        if not retrieved_chunk_ids:
            missing_retrieval += 1

        p5 = precision_at_k(retrieved_chunk_ids, gold_chunk_ids, k=top_k)
        r5 = recall_at_k(retrieved_chunk_ids, gold_chunk_ids, k=top_k)
        hits = len(set(retrieved_chunk_ids[:top_k]) & set(gold_chunk_ids))
        hit_at_5 = 1 if hits > 0 else 0

        row = {
            "question_id": qid,
            "system": run_name,
            "num_gold_chunks": len(gold_chunk_ids),
            "num_retrieved_chunks": len(retrieved_chunk_ids),
            "hits_at_5": hits,
            "hit_at_5": hit_at_5,
            "precision_at_5": p5,
            "recall_at_5": r5,
            "gold_chunk_ids": gold_chunk_ids,
            "retrieved_chunk_ids_top5": retrieved_chunk_ids[:top_k],
            "retrieved_chunk_ids_all": retrieved_chunk_ids,
        }

        # agentic-specific metrics
        if run_name.startswith("agentic_rag"):
            first_round_ids = extract_agentic_first_round(rec)
            second_round_ids = extract_agentic_second_round(rec)
            final_evidence_ids = extract_agentic_final_evidence(rec)

            standard_rec = standard_records_by_qid.get(qid, {})
            standard_ids = extract_chunk_id_list_from_standard_record(standard_rec)
            standard_topk = standard_ids[:top_k]

            row.update({
                "first_round_top5": first_round_ids[:top_k],
                "second_round_top5": second_round_ids[:top_k],
                "final_evidence_count": len(final_evidence_ids),
                "used_second_round": bool(rec.get("used_second_round", False)),
                "sufficiency_judgment": rec.get("sufficiency_judgment"),
                "recall_at_all_evidence": recall_all_evidence(final_evidence_ids, gold_chunk_ids),
                "gold_gain_over_standard_top5": gold_gain_over_standard(
                    standard_topk, final_evidence_ids, gold_chunk_ids
                ),
                "extra_gold_coverage_vs_standard_top5": extra_gold_coverage(
                    standard_topk, final_evidence_ids, gold_chunk_ids
                ),
                "new_chunks_beyond_standard_top5": [
                    x for x in final_evidence_ids if x not in set(standard_topk)
                ],
            })

        per_question_rows.append(row)

    if per_question_rows:
        avg_p5 = sum(r["precision_at_5"] for r in per_question_rows) / len(per_question_rows)
        avg_r5 = sum(r["recall_at_5"] for r in per_question_rows) / len(per_question_rows)
        avg_hits_at_5 = sum(r["hits_at_5"] for r in per_question_rows) / len(per_question_rows)
        hit_at_5_rate = sum(r["hit_at_5"] for r in per_question_rows) / len(per_question_rows)
    else:
        avg_p5 = 0.0
        avg_r5 = 0.0
        avg_hits_at_5 = 0.0
        hit_at_5_rate = 0.0

    summary = {
        "system": run_name,
        "num_records": len(records),
        "num_evaluated": len(per_question_rows),
        "missing_question_ids": missing_qids,
        "records_without_retrieval_info": missing_retrieval,
        "precision_at_5": avg_p5,
        "recall_at_5": avg_r5,
        "avg_hits_at_5": avg_hits_at_5,
        "hit_at_5": hit_at_5_rate,
    }

    if run_name.startswith("agentic_rag") and per_question_rows:
        summary.update({
            "recall_at_all_evidence": sum(r["recall_at_all_evidence"] for r in per_question_rows) / len(per_question_rows),
            "avg_gold_gain_over_standard_top5": sum(r["gold_gain_over_standard_top5"] for r in per_question_rows) / len(per_question_rows),
            "avg_extra_gold_coverage_vs_standard_top5": sum(r["extra_gold_coverage_vs_standard_top5"] for r in per_question_rows) / len(per_question_rows),
            "avg_final_evidence_count": sum(r["final_evidence_count"] for r in per_question_rows) / len(per_question_rows),
            "second_round_usage_rate": sum(1 for r in per_question_rows if r["used_second_round"]) / len(per_question_rows),
        })

    return per_question_rows, summary


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    questions = load_json(QUESTIONS_PATH)
    fixed_chunks = load_jsonl(FIXED_CHUNKS_PATH)
    recursive_chunks = load_jsonl(RECURSIVE_CHUNKS_PATH)

    questions_by_id = {q["question_id"]: q for q in questions}
    fixed_doc_to_chunks = build_doc_to_chunks(fixed_chunks)
    recursive_doc_to_chunks = build_doc_to_chunks(recursive_chunks)

    # load standard runs so agentic can compare against corresponding standard top5
    standard_records_by_run_and_qid = {}
    for run_name, run_file in RUN_FILES.items():
        if run_name.startswith("standard_rag") and Path(run_file).exists():
            rows = load_jsonl(run_file)
            standard_records_by_run_and_qid[run_name] = {
                r["question_id"]: r for r in rows if "question_id" in r
            }

    all_rows = []
    all_summaries = []

    for run_name, run_file in RUN_FILES.items():
        if not Path(run_file).exists():
            print(f"[SKIP] Missing run file: {run_file}")
            continue

        if run_name == "agentic_rag_fixed":
            standard_ref = standard_records_by_run_and_qid.get("standard_rag_fixed", {})
        elif run_name == "agentic_rag_recursive":
            standard_ref = standard_records_by_run_and_qid.get("standard_rag_recursive", {})
        else:
            standard_ref = {}

        print(f"Evaluating {run_name} ...")
        rows, summary = evaluate_run(
            run_name=run_name,
            run_file=run_file,
            questions_by_id=questions_by_id,
            fixed_doc_to_chunks=fixed_doc_to_chunks,
            recursive_doc_to_chunks=recursive_doc_to_chunks,
            standard_records_by_qid=standard_ref,
            top_k=TOP_K,
        )
        all_rows.extend(rows)
        all_summaries.append(summary)

    save_jsonl("retrieval_eval_per_question.jsonl", all_rows)
    save_json("retrieval_eval_summary.json", all_summaries)

    print("\nDone.")
    print("Saved per-question results to retrieval_eval_per_question.jsonl")
    print("Saved summary results to retrieval_eval_summary.json")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
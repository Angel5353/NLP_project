from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Basic I/O
# =========================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Small utility helpers
# =========================

def first_present(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Return the first present key from a dict.
    """
    for k in keys:
        if k in d:
            return d[k]
    return default


def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def safe_read_text(path: Path) -> str:
    """
    Read text with a couple of reasonable fallbacks.
    """
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Failed to decode {path}")


def normalize_relative_path(path_str: str) -> str:
    """
    Normalize a path to a forward-slash relative style.
    """
    return str(Path(path_str)).replace("\\", "/")


# =========================
# Corpus preparation
# =========================

def build_corpus(corpus_dir: str) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Scan all .txt files under corpus_dir recursively.

    Returns:
        corpus_records: list of {"doc_id": ..., "text": ...}
        doc_lookup: mapping from possible path aliases to canonical doc_id
    """
    corpus_root = Path(corpus_dir)
    if not corpus_root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    txt_files = sorted(corpus_root.rglob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found under corpus directory: {corpus_dir}")

    corpus_records: List[Dict[str, str]] = []
    doc_lookup: Dict[str, str] = {}

    for file_path in txt_files:
        rel_path = file_path.relative_to(corpus_root)
        rel_str = normalize_relative_path(str(rel_path))
        rel_no_suffix = normalize_relative_path(str(rel_path.with_suffix("")))

        text = safe_read_text(file_path)

        # Canonical doc_id:
        # keep relative path without .txt
        doc_id = rel_no_suffix

        corpus_records.append({
            "doc_id": doc_id,
            "text": text,
        })

        # Add a few aliases so benchmark references are easier to resolve later
        doc_lookup[rel_str] = doc_id
        doc_lookup[rel_no_suffix] = doc_id
        doc_lookup[file_path.name] = doc_id
        doc_lookup[file_path.stem] = doc_id

    return corpus_records, doc_lookup


# =========================
# Benchmark field extraction
# =========================

def extract_question_text(case: Dict[str, Any]) -> Optional[str]:
    """
    Try common field names for the benchmark query.
    """
    value = first_present(
        case,
        ["query", "question", "prompt", "input", "text"],
        default=None,
    )
    if isinstance(value, str):
        return value.strip()
    return None


def extract_case_id(case: Dict[str, Any], fallback_id: str) -> str:
    """
    Try common field names for a stable question id.
    """
    value = first_present(
        case,
        ["id", "question_id", "query_id", "uid", "uuid"],
        default=None,
    )
    if value is None:
        return fallback_id
    return str(value)


def extract_ground_truth_list(case: Dict[str, Any]) -> List[Any]:
    """
    Try common names for the gold evidence list.
    README says each test case has a ground truth array of snippets.
    """
    gt = first_present(
        case,
        ["ground_truth", "snippets", "evidence", "answers", "gold_evidence"],
        default=[],
    )
    return ensure_list(gt)


def extract_file_path(snippet: Dict[str, Any]) -> Optional[str]:
    """
    Try common names for the corpus file reference.
    """
    value = first_present(
        snippet,
        [
            "file_path",
            "filepath",
            "path",
            "file",
            "source_file",
            "document_path",
            "document",
            "doc_path",
            "corpus_path",
        ],
        default=None,
    )

    if isinstance(value, str):
        return normalize_relative_path(value.strip())
    return None


def extract_start_end(snippet: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Try common names for character start / end offsets.
    Supports either:
        - start_char / end_char
        - span = [start, end]
    """
    if "span" in snippet and isinstance(snippet["span"], list) and len(snippet["span"]) == 2:
        try:
            start = int(snippet["span"][0])
            end = int(snippet["span"][1])
            return start, end
        except (TypeError, ValueError):
            pass

    start = first_present(
        snippet,
        ["start_char", "char_start", "start", "begin", "offset_start"],
        default=None,
    )
    end = first_present(
        snippet,
        ["end_char", "char_end", "end", "stop", "offset_end"],
        default=None,
    )

    try:
        start = int(start) if start is not None else None
    except (TypeError, ValueError):
        start = None

    try:
        end = int(end) if end is not None else None
    except (TypeError, ValueError):
        end = None

    return start, end


def extract_snippet_text(snippet: Dict[str, Any]) -> Optional[str]:
    """
    If raw snippet text is already stored in the benchmark, use it.
    """
    value = first_present(
        snippet,
        ["text", "snippet", "content", "evidence_text", "quote", "answer"],
        default=None,
    )
    if isinstance(value, str):
        return value
    return None


# =========================
# Benchmark -> project format conversion
# =========================

def resolve_doc_id(file_path: Optional[str], doc_lookup: Dict[str, str]) -> Optional[str]:
    """
    Map a benchmark file path to canonical corpus doc_id.
    """
    if not file_path:
        return None

    normalized = normalize_relative_path(file_path)

    if normalized in doc_lookup:
        return doc_lookup[normalized]

    # try no suffix
    no_suffix = normalize_relative_path(str(Path(normalized).with_suffix("")))
    if no_suffix in doc_lookup:
        return doc_lookup[no_suffix]

    # try basename
    basename = Path(normalized).name
    if basename in doc_lookup:
        return doc_lookup[basename]

    # try stem
    stem = Path(normalized).stem
    if stem in doc_lookup:
        return doc_lookup[stem]

    return None


def rebuild_text_from_offsets(
    doc_id: Optional[str],
    start_char: Optional[int],
    end_char: Optional[int],
    corpus_text_by_doc_id: Dict[str, str],
) -> Optional[str]:
    """
    Recover snippet text from corpus document text if offsets are available.
    """
    if doc_id is None or start_char is None or end_char is None:
        return None

    doc_text = corpus_text_by_doc_id.get(doc_id)
    if doc_text is None:
        return None

    if start_char < 0 or end_char < 0 or start_char >= end_char:
        return None

    if end_char > len(doc_text):
        end_char = len(doc_text)

    return doc_text[start_char:end_char]


def convert_case(
    case: Dict[str, Any],
    source_benchmark: str,
    fallback_case_id: str,
    doc_lookup: Dict[str, str],
    corpus_text_by_doc_id: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    Convert one raw benchmark case into your project's unified question format.
    """
    question = extract_question_text(case)
    if not question:
        return None

    question_id = extract_case_id(
        case,
        fallback_id=f"{source_benchmark}_{fallback_case_id}",
    )

    raw_snippets = extract_ground_truth_list(case)
    gold_evidence: List[Dict[str, Any]] = []

    for snippet in raw_snippets:
        if not isinstance(snippet, dict):
            continue

        raw_file_path = extract_file_path(snippet)
        doc_id = resolve_doc_id(raw_file_path, doc_lookup)

        start_char, end_char = extract_start_end(snippet)

        snippet_text = extract_snippet_text(snippet)
        if snippet_text is None:
            snippet_text = rebuild_text_from_offsets(
                doc_id=doc_id,
                start_char=start_char,
                end_char=end_char,
                corpus_text_by_doc_id=corpus_text_by_doc_id,
            )

        gold_evidence.append({
            "doc_id": doc_id,
            "raw_file_path": raw_file_path,
            "start_char": start_char,
            "end_char": end_char,
            "text": snippet_text,
        })

    return {
        "question_id": str(question_id),
        "question": question,
        "source_benchmark": source_benchmark,
        "gold_evidence": gold_evidence,
    }


def find_case_list(obj):
    """
    Try to find the benchmark case list from a nested JSON object.
    Returns a list if found, otherwise [].
    """
    if isinstance(obj, list):
        return obj

    if not isinstance(obj, dict):
        return []

    # 1) common direct keys
    for key in ["tests", "test_cases", "cases", "data", "examples", "items", "queries"]:
        if key in obj and isinstance(obj[key], list):
            return obj[key]

    # 2) nested one level down
    for key, value in obj.items():
        if isinstance(value, dict):
            for subkey in ["tests", "test_cases", "cases", "data", "examples", "items", "queries"]:
                if subkey in value and isinstance(value[subkey], list):
                    return value[subkey]

    # 3) if any top-level value itself is a list, use the first plausible one
    for key, value in obj.items():
        if isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict):
                return value

    # 4) recursive fallback
    def recursive_search(x):
        if isinstance(x, list):
            if len(x) == 0:
                return []
            if isinstance(x[0], dict):
                return x
            return []

        if isinstance(x, dict):
            for _, v in x.items():
                found = recursive_search(v)
                if found:
                    return found

        return []

    return recursive_search(obj)


def convert_benchmark_file(
    benchmark_path: Path,
    doc_lookup: Dict[str, str],
    corpus_text_by_doc_id: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = load_json(str(benchmark_path))
    source_benchmark = benchmark_path.stem

    cases = find_case_list(raw)

    converted: List[Dict[str, Any]] = []
    skipped = 0

    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            skipped += 1
            continue

        item = convert_case(
            case=case,
            source_benchmark=source_benchmark,
            fallback_case_id=str(i),
            doc_lookup=doc_lookup,
            corpus_text_by_doc_id=corpus_text_by_doc_id,
        )
        if item is None:
            skipped += 1
            continue
        converted.append(item)

    stats = {
        "benchmark_file": str(benchmark_path),
        "source_benchmark": source_benchmark,
        "num_raw_cases": len(cases),
        "num_converted_cases": len(converted),
        "num_skipped_cases": skipped,
    }

    return converted, stats


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LegalBench-RAG data into corpus.json and questions.json"
    )

    parser.add_argument(
        "--corpus_dir",
        type=str,
        required=True,
        help="Path to raw corpus directory (e.g. data/raw/corpus)",
    )
    parser.add_argument(
        "--benchmarks_dir",
        type=str,
        required=True,
        help="Path to raw benchmarks directory (e.g. data/raw/benchmarks)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed outputs",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Number of examples to save in sample_questions.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build processed corpus
    print("Building corpus.json from raw corpus...")
    corpus_records, doc_lookup = build_corpus(args.corpus_dir)
    corpus_text_by_doc_id = {x["doc_id"]: x["text"] for x in corpus_records}

    corpus_path = output_dir / "corpus.json"
    save_json(str(corpus_path), corpus_records)
    print(f"Saved corpus.json to {corpus_path}")

    # 2) Convert benchmark files
    benchmarks_root = Path(args.benchmarks_dir)
    if not benchmarks_root.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {args.benchmarks_dir}")

    benchmark_files = sorted(benchmarks_root.rglob("*.json"))
    if not benchmark_files:
        raise ValueError(f"No benchmark .json files found under: {args.benchmarks_dir}")

    print(f"Found {len(benchmark_files)} benchmark files.")

    all_questions: List[Dict[str, Any]] = []
    benchmark_stats: List[Dict[str, Any]] = []

    for benchmark_file in benchmark_files:
        print(f"Processing {benchmark_file} ...")
        converted, stats = convert_benchmark_file(
            benchmark_path=benchmark_file,
            doc_lookup=doc_lookup,
            corpus_text_by_doc_id=corpus_text_by_doc_id,
        )
        all_questions.extend(converted)
        benchmark_stats.append(stats)

    # 3) Save full questions
    questions_path = output_dir / "questions.json"
    save_json(str(questions_path), all_questions)
    print(f"Saved questions.json to {questions_path}")

    # 4) Save a small sample for debugging
    random.seed(args.seed)
    if len(all_questions) <= args.sample_size:
        sample_questions = all_questions
    else:
        sample_questions = random.sample(all_questions, args.sample_size)

    sample_questions_path = output_dir / "sample_questions.json"
    save_json(str(sample_questions_path), sample_questions)
    print(f"Saved sample_questions.json to {sample_questions_path}")

    # 5) Save summary
    unresolved_doc_refs = 0
    total_gold_evidence = 0
    missing_offsets = 0
    missing_text = 0

    for item in all_questions:
        for ev in item.get("gold_evidence", []):
            total_gold_evidence += 1
            if ev.get("doc_id") is None:
                unresolved_doc_refs += 1
            if ev.get("start_char") is None or ev.get("end_char") is None:
                missing_offsets += 1
            if ev.get("text") in (None, ""):
                missing_text += 1

    summary = {
        "num_documents": len(corpus_records),
        "num_questions": len(all_questions),
        "num_benchmark_files": len(benchmark_files),
        "sample_size": len(sample_questions),
        "benchmark_stats": benchmark_stats,
        "gold_evidence_stats": {
            "total_gold_evidence": total_gold_evidence,
            "unresolved_doc_refs": unresolved_doc_refs,
            "missing_offsets": missing_offsets,
            "missing_text": missing_text,
        },
        "output_files": {
            "corpus_json": str(corpus_path),
            "questions_json": str(questions_path),
            "sample_questions_json": str(sample_questions_path),
        },
    }

    summary_path = output_dir / "prepare_data_summary.json"
    save_json(str(summary_path), summary)
    print(f"Saved summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
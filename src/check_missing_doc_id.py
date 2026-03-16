import json

with open("data/processed/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

bad = []
for item in questions:
    for ev in item.get("gold_evidence", []):
        if ev.get("doc_id") is None:
            bad.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "source_benchmark": item["source_benchmark"],
                "raw_file_path": ev.get("raw_file_path"),
                "start_char": ev.get("start_char"),
                "end_char": ev.get("end_char"),
                "text_preview": (ev.get("text") or "")[:200],
            })

print("num unresolved:", len(bad))
for x in bad:
    print("=" * 80)
    print(x)
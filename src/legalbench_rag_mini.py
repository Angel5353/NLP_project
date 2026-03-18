import json
import random
from pathlib import Path
from collections import defaultdict

INPUT_PATH = Path("data/processed/questions.json")
OUTPUT_PATH = Path("data/processed/questions_mini.json")

TARGET_BENCHMARKS = ["privacy_qa", "cuad", "maud", "contractnli"]
N_PER_BENCHMARK = 194
RANDOM_SEED = 42


def get_source_name(item: dict) -> str:
    """
    Robustly get benchmark/source name from a processed question item.
    Adjust this if your field names differ.
    """
    for key in ["source_benchmark", "benchmark", "dataset", "source"]:
        if key in item:
            return str(item[key]).strip().lower()
    raise KeyError(f"Cannot find benchmark/source field in item: {item.keys()}")


def main():
    random.seed(RANDOM_SEED)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    grouped = defaultdict(list)
    for q in questions:
        source = get_source_name(q)
        grouped[source].append(q)

    mini_questions = []

    for benchmark in TARGET_BENCHMARKS:
        group = grouped.get(benchmark, [])
        if len(group) < N_PER_BENCHMARK:
            raise ValueError(
                f"Benchmark '{benchmark}' only has {len(group)} questions, "
                f"but {N_PER_BENCHMARK} are required."
            )

        sampled = random.sample(group, N_PER_BENCHMARK)
        mini_questions.extend(sampled)

    # Optional: shuffle overall output so it isn't grouped by benchmark
    random.shuffle(mini_questions)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mini_questions, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(mini_questions)} questions to {OUTPUT_PATH}")
    for benchmark in TARGET_BENCHMARKS:
        print(f"{benchmark}: {N_PER_BENCHMARK}")


if __name__ == "__main__":
    main()
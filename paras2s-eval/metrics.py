import re
import json
from pathlib import Path

batch_request_results_path = Path("/work/u3937558/StyleTalk/batch_request_results.jsonl")

results = open(batch_request_results_path, "r").readlines()
results = [json.loads(line) for line in results]
continuation_fitness_scores = []

for res in results:
    try:
        match = re.search(r"The score is (\d+)", res["response"]["body"]["choices"][0]["message"]["content"])
        score = match.group(1)
        continuation_fitness_scores.append(int(score))
    except Exception as e:
        print(f"Error while processing item {res}: {e}")

print(continuation_fitness_scores)
print(sum(continuation_fitness_scores) / len(continuation_fitness_scores))

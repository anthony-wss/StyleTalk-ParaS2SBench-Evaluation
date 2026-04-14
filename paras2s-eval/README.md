1. Generate Audio-Reasoner for model's response tone.

```
bash run-inference.sh
```

2. Generate transcription for model's response.

```
bash run-whisper.sh
```

3. Generate the batch request jsonl to use gpt-4.1 to give the continuation fitness score (1-5).

```
bash run-gen-batch-request.sh
```

4. Submit the request manually and get the results when it's done.

5. Compute the continuation fitness metrics score.

```
bash run-paras2s-metrics.sh
```

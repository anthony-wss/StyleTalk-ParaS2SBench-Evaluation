import os
from typing import List, Literal
import re
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset, get_template
from swift.plugin import InferStats
from pathlib import Path
import json


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=2048, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']
    output = ""
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        if resp_list[0] is None:
            continue
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
        output += resp_list[0].choices[0].delta.content
    print()
    print(f'metric: {metric.compute()}')
    return output


def get_message(audiopath, prompt):
    messages = [
        {"role": "system", "content": system},
        {
        'role':
        'user',
        'content': [{
            'type': 'audio',
            'audio': audiopath
        }, {
            'type': 'text',
            'text':  prompt
        }]
    }]
    return messages

system = 'You are an audio deep-thinking model. Upon receiving a question, please respond in two parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.'
infer_backend = 'pt'
model = 'qwen2_audio'
last_model_checkpoint = "/work/u3937558/.cache/huggingface/hub/models--zhifeixie--Audio-Reasoner/snapshots/f38198da84d4e02623a83fa1d005ad31f1d6a6a7" #Please replace it with the path to checkpoint
engine = PtEngine(last_model_checkpoint, max_batch_size=64,  model_type = model)

def audioreasoner_gen(audiopath, prompt):
    return infer_stream(engine, InferRequest(messages=get_message(audiopath, prompt)))

def main():
    source_audio_dir = Path("/work/u3937558/SLAM-LLM/exp/s2s_train_v4-Qwen2-0.5b-gpu4-btz3-lr1e-4-fp16-epochs10-whisper_small-latency0-group3/s2s_epoch_3_step_19594/s2s_decode__trp1.2_arp1.2_seed777_greedy/pred_audio/prompt_6")
    output_file = Path("/work/u3937558/StyleTalk/output")

    audio_list = sorted(os.listdir(source_audio_dir))

    prompt = """You are an expert speech and audio analyst. Your task is to listen to the provided spoken response and classify its speaking style across three dimensions: emotion, speed, and volume. 

You MUST strictly use ONLY the predefined categories listed below. Do not use any other words or variations.

[Categories]
1. Emotion: "neutral", "cheerful", "hopeful", "sad", "friendly", or "unfriendly".
2. Speed: "slow", "normal", or "fast".
3. Volume: "quiet", "normal", or "loud".

[Instructions]
1. Carefully analyze the acoustic features, prosody, and tone of the speaker in the audio.
2. Select the single most appropriate label for each of the three dimensions.
3. Output the result strictly in JSON format without any markdown blocks or additional explanation.

[Expected Output Format]
{
  "emotion": "<selected_emotion>",
  "speed": "<selected_speed>",
  "volume": "<selected_volume>"
}"""

    with open(output_file, "w") as f:
        for audio_file in audio_list:
            audiopath = source_audio_dir / audio_file
            index = Path(audio_file).stem
            output = audioreasoner_gen(audiopath, prompt)
            print(json.dumps({"key": index,"output": output}), file=f, flush=True)
   

if __name__ == '__main__':
    main()

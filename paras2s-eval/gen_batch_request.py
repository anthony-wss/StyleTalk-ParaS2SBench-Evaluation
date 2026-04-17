from pathlib import Path
import json
import re
import pandas as pd
from datetime import datetime

model_response_tone_path = Path("/work/u3937558/StyleTalk/data/mini_omni_output_tone")
model_response_trans_path = Path("/work/u3937558/StyleTalk/data/mini_omni_transcription")
styletalk_eval_path = Path("/work/u3937558/StyleTalk/eval.csv")
judge_prompt_template_path = Path("/work/u3937558/StyleTalk/paras2sbench_prompt.txt")
batch_request_output_path = Path("/work/u3937558/StyleTalk/data/mini_omni_batch_request_output.jsonl")

if __name__ == "__main__":
    # Initialize all the data dict
    context = {}
    user_inp_trans = {}
    user_inp_emo = {}
    user_inp_speed = {}
    user_inp_volume = {}
    model_res_tone = {}
    model_res_trans = {}

    # Extract model's response tone by Audio-Reasoner
    model_output = open(model_response_tone_path, "r").readlines()
    for out in model_output:
        out = json.loads(out)
        key = out["key"]
        match = re.search(f"<RESPONSE>(.*?)</RESPONSE>", out["output"], re.DOTALL)
        if match:
            try:
                model_res_tone[key] = match.group(1).strip()
            except Exception as e:
                print(f"Error: The response for key {key} ran into error: {e}")
    
    # Load model's response transcription
    model_output = open(model_response_trans_path, "r").readlines()
    for out in model_output:
        out = json.loads(out)
        key = out["key"]
        model_res_trans[key] = out["output"]

    # Load user input's transcription and context
    # We will load the first `len(model_res_trans)` rows
    eval_df = pd.read_csv(styletalk_eval_path)
    for i in range(len(model_res_trans)):
        user_inp_trans[str(i+1)]  = eval_df.at[i, "curr_text"]
        user_inp_emo[str(i+1)]    = eval_df.at[i, "curr_emotion"]
        user_inp_speed[str(i+1)]  = eval_df.at[i, "curr_speed"]
        user_inp_volume[str(i+1)] = eval_df.at[i, "curr_volume"]
        context[str(i+1)]         = eval_df.at[i, "context"]
    
    # Generate the batch request jsonl for continuation fitness score
    prompt_template = "".join(open(judge_prompt_template_path, "r").readlines())

    timestring = datetime.now().strftime(r"%Y%m%d%H%M%S")
    with open(batch_request_output_path, "w") as f:
        for i in range(len(model_res_trans)):
            key = str(i+1)

            try:
                prompt = prompt_template\
                    .replace("CONTEXT", context[key])\
                    .replace("USER_TRANSCRIPTION", user_inp_trans[key])\
                    .replace("USER_EMOTION", user_inp_emo[key])\
                    .replace("USER_SPEED", user_inp_speed[key])\
                    .replace("USER_VOLUME", user_inp_volume[key])\
                    .replace("MODEL_RES_TRANSCRIPTION", model_res_trans[key])\
                    .replace("MODEL_RES_TONE", model_res_tone[key])
            except KeyError as e:
                print(f"Sample with key {key} is skipped because transcription or tone missing.")

            print(json.dumps({
                "custom_id": f"request-{timestring}-{key}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                "max_tokens": 4096}}), file=f, flush=True)

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
import os
import json
from tqdm import tqdm

source_audio_dir = Path("/work/u3937558/SLAM-LLM/exp/s2s_train_v4-Qwen2-0.5b-gpu2-btz3-lr1e-4-fp16-epochs10-whisper_small-latency0-group1/s2s_epoch_2_step_45797/s2s_decode__trp1.2_arp1.2_seed777_greedy/pred_audio/prompt_6")
output_file = Path("/work/u3937558/StyleTalk/data/mini_omni_transcription")

device = "cuda:0"
dtype = torch.float16

model_id = "openai/whisper-large-v3"

if __name__ == "__main__":
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device=device,
    )

    audio_list = sorted(os.listdir(source_audio_dir))
    
    with open(output_file, "w") as f:
        for audio_file in tqdm(audio_list):
            audiopath = source_audio_dir / audio_file
            index = Path(audio_file).stem
            result = pipe(str(audiopath), return_timestamps=True, language='en')

            print(json.dumps({"key": index, "output": result["text"]}), file=f, flush=True)

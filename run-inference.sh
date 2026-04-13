#/bin/bash

srun --partition dev --gpus-per-node 1 --account MST115022 -N 1 -n 1 \
    singularity exec --nv -B /work/u3937558:/work/u3937558 \
  /work/u3937558/SLAM-LLM/slam-omni.sif \
  bash -c "\
  source /work/u3937558/StyleTalk/.venv/bin/activate && python /work/u3937558/StyleTalk/Audio-Reasoner/inference.py"

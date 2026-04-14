#/bin/bash
export OPENBLAS_NUM_THREADS=1

singularity exec --nv -B /work/u3937558:/work/u3937558 \
  /work/u3937558/SLAM-LLM/slam-omni.sif \
  bash -c "\
  source /work/u3937558/StyleTalk/.venv/bin/activate && python /work/u3937558/StyleTalk/paras2s-eval/gen_batch_request.py"

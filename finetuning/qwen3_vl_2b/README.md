# Qwen3-VL 2B fine-tuning

This directory contains the first public training component of SurgLQA: a
DeepSpeed + LoRA supervised fine-tuning recipe for
`Qwen/Qwen3-VL-2B-Instruct`.

The dataset, checkpoints, evaluation code, and the remaining SurgLQA pipeline
are not included in this release. Their status is tracked in the repository
root README.

## Installation

Python 3.10 or 3.11 and CUDA-capable GPUs are recommended.

```bash
cd finetuning/qwen3_vl_2b
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Dataset format

The annotation file is a JSON array or JSONL file. Each record follows the
Qwen-VL conversation format:

```json
{
  "video": "relative/path/to/video.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nWhat is happening in the surgical video?"
    },
    {
      "from": "gpt",
      "value": "The response for supervised fine-tuning."
    }
  ]
}
```

Use `image` with an `<image>` token for image samples. Relative media paths are
resolved under `SURGLQA_MEDIA_ROOT`.

## Training

```bash
export SURGLQA_ANNOTATION=/path/to/surglqa_train.json
export SURGLQA_MEDIA_ROOT=/path/to/surglqa_media
bash scripts/train_lora.sh
```

Optional environment variables:

- `MODEL_PATH`: Hugging Face model ID or local model path; defaults to
  `Qwen/Qwen3-VL-2B-Instruct`.
- `OUTPUT_DIR`: checkpoint directory.
- `CACHE_DIR`: model cache directory.
- `NPROC_PER_NODE`: GPU count; otherwise detected with `nvidia-smi`.
- `MASTER_ADDR` and `MASTER_PORT`: distributed launch settings.

Extra command-line arguments are forwarded to the trainer, so settings can be
overridden without editing the script, for example:

```bash
bash scripts/train_lora.sh --per_device_train_batch_size 2 --save_steps 500
```

## Data validation

The bundled validator verifies media paths and `<image>`/`<video>` token counts:

```bash
python tools/check_image.py "$SURGLQA_ANNOTATION" "$SURGLQA_MEDIA_ROOT"
```

## Attribution

The trainer is adapted from the Qwen3-VL fine-tuning framework at upstream
commit `9658872`. The copied framework files retain their Apache-2.0 license;
see `LICENSE` in this directory and the
[upstream training documentation](https://github.com/QwenLM/Qwen3-VL/tree/9658872/qwen-vl-finetune).

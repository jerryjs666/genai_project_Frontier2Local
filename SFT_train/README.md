# SFT_train

This folder contains the supervised fine-tuning (SFT) setup and results for training a LoRA adapter on GSM8K-style math reasoning data.

## What Is Inside

- `qwen25_3b_llamafactory_lora_sft_colab_train.ipynb`: Colab notebook used to run the training workflow.
- `train_qwen25_3b_lora_sft.yaml`: LLaMA-Factory training configuration.
- `data/`: Training and validation data in LLaMA-Factory dataset format.
- `outputs/qwen25_3b_gsm8k_lora_sft_full/`: Final LoRA adapter, tokenizer files, logs, plots, and training metrics.
- `base_model_val_results.json`: Validation results for the base model.
- `val_results.json`: Validation results after SFT.

## Model And Method

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Fine-tuning method: LoRA
- Training framework: LLaMA-Factory / PEFT / Transformers
- Task: GSM8K math problem solving
- Main metric tracked during training: evaluation loss

The LoRA configuration is defined in `train_qwen25_3b_lora_sft.yaml`:

- LoRA rank: `32`
- LoRA alpha: `64`
- LoRA dropout: `0.05`
- Target modules: attention and MLP projection layers
- Epochs: `4`
- Learning rate: `2e-5`
- Scheduler: cosine
- Batch size per device: `16`

## Data

The `data/` folder contains three dataset files:

- `gsm8k_sft_all.json`: full SFT dataset
- `gsm8k_sft_train.json`: training split
- `gsm8k_sft_val.json`: validation split

`dataset_info.json` maps these files into the format expected by LLaMA-Factory:

- `instruction` -> prompt
- `input` -> query
- `output` -> response

## Training

The easiest way to reproduce the run is to open:

```text
qwen25_3b_llamafactory_lora_sft_colab_train.ipynb
```

The notebook uses the YAML config:

```text
train_qwen25_3b_lora_sft.yaml
```

The config currently assumes the project is placed at:

```text
/content/project/SFT_train
```

If the folder is moved, update `dataset_dir` and `output_dir` in the YAML file before training.

## Results

The final training output is stored in:

```text
outputs/qwen25_3b_gsm8k_lora_sft_full/
```

Key results from the saved trainer metrics:

- Final eval loss: `0.2762`
- Train loss: `0.2601`
- Training runtime: about `2419.6` seconds
- Epochs: `4`

Useful output files:

- `adapter_model.safetensors`: trained LoRA adapter weights
- `adapter_config.json`: LoRA adapter configuration
- `trainer_log.jsonl`: step-by-step training log
- `trainer_state.json`: trainer state and best checkpoint information
- `training_loss.png`: training loss curve
- `training_eval_loss.png`: evaluation loss curve
- `all_results.json`, `train_results.json`, `eval_results.json`: summary metrics

## Notes For Teammates

- Use the notebook for the full Colab workflow.
- Use the YAML file to inspect or modify hyperparameters.
- Use `val_results.json` and `base_model_val_results.json` to compare the SFT model against the base model.
- The output adapter is a LoRA adapter, not a full merged model.

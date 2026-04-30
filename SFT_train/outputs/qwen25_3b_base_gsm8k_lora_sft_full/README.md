---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-3B
tags:
- base_model:adapter:Qwen/Qwen2.5-3B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen25_3b_base_gsm8k_lora_sft_full
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen25_3b_base_gsm8k_lora_sft_full

This model is a fine-tuned version of [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) on the gsm8k_sft_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3695

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.8145        | 0.0611 | 25   | 0.7540          |
| 0.5709        | 0.1222 | 50   | 0.5054          |
| 0.4661        | 0.1834 | 75   | 0.4497          |
| 0.4259        | 0.2445 | 100  | 0.4316          |
| 0.4317        | 0.3056 | 125  | 0.4203          |
| 0.3963        | 0.3667 | 150  | 0.4124          |
| 0.4290        | 0.4279 | 175  | 0.4067          |
| 0.4047        | 0.4890 | 200  | 0.4018          |
| 0.3992        | 0.5501 | 225  | 0.3980          |
| 0.4093        | 0.6112 | 250  | 0.3947          |
| 0.4050        | 0.6724 | 275  | 0.3929          |
| 0.3906        | 0.7335 | 300  | 0.3896          |
| 0.3734        | 0.7946 | 325  | 0.3869          |
| 0.3820        | 0.8557 | 350  | 0.3854          |
| 0.3793        | 0.9169 | 375  | 0.3838          |
| 0.4031        | 0.9780 | 400  | 0.3819          |
| 0.3796        | 1.0391 | 425  | 0.3814          |
| 0.3654        | 1.1002 | 450  | 0.3804          |
| 0.3709        | 1.1614 | 475  | 0.3792          |
| 0.3520        | 1.2225 | 500  | 0.3775          |
| 0.3543        | 1.2836 | 525  | 0.3777          |
| 0.3536        | 1.3447 | 550  | 0.3764          |
| 0.3683        | 1.4059 | 575  | 0.3764          |
| 0.3734        | 1.4670 | 600  | 0.3753          |
| 0.3895        | 1.5281 | 625  | 0.3746          |
| 0.3599        | 1.5892 | 650  | 0.3747          |
| 0.3684        | 1.6504 | 675  | 0.3727          |
| 0.3489        | 1.7115 | 700  | 0.3726          |
| 0.3570        | 1.7726 | 725  | 0.3721          |
| 0.3441        | 1.8337 | 750  | 0.3719          |
| 0.3568        | 1.8949 | 775  | 0.3717          |
| 0.3693        | 1.9560 | 800  | 0.3709          |
| 0.3463        | 2.0171 | 825  | 0.3709          |
| 0.3327        | 2.0782 | 850  | 0.3714          |
| 0.3684        | 2.1394 | 875  | 0.3706          |
| 0.3445        | 2.2005 | 900  | 0.3704          |
| 0.3370        | 2.2616 | 925  | 0.3703          |
| 0.3479        | 2.3227 | 950  | 0.3703          |
| 0.3437        | 2.3839 | 975  | 0.3703          |
| 0.3335        | 2.4450 | 1000 | 0.3699          |
| 0.3478        | 2.5061 | 1025 | 0.3698          |
| 0.3390        | 2.5672 | 1050 | 0.3697          |
| 0.3459        | 2.6284 | 1075 | 0.3698          |
| 0.3463        | 2.6895 | 1100 | 0.3696          |
| 0.3362        | 2.7506 | 1125 | 0.3697          |
| 0.3605        | 2.8117 | 1150 | 0.3697          |
| 0.3602        | 2.8729 | 1175 | 0.3696          |
| 0.3304        | 2.9340 | 1200 | 0.3695          |
| 0.3646        | 2.9951 | 1225 | 0.3696          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.0.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2
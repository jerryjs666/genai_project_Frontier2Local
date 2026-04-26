---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen25_3b_gsm8k_lora_sft_full
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen25_3b_gsm8k_lora_sft_full

This model is a fine-tuned version of [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) on the gsm8k_sft_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2762

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 4.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.9279        | 0.0611 | 25   | 0.6485          |
| 0.4520        | 0.1222 | 50   | 0.3899          |
| 0.3570        | 0.1834 | 75   | 0.3415          |
| 0.3185        | 0.2445 | 100  | 0.3246          |
| 0.3258        | 0.3056 | 125  | 0.3137          |
| 0.2929        | 0.3667 | 150  | 0.3069          |
| 0.3275        | 0.4279 | 175  | 0.3022          |
| 0.3061        | 0.4890 | 200  | 0.2996          |
| 0.2964        | 0.5501 | 225  | 0.2960          |
| 0.3068        | 0.6112 | 250  | 0.2946          |
| 0.3049        | 0.6724 | 275  | 0.2934          |
| 0.2910        | 0.7335 | 300  | 0.2904          |
| 0.2759        | 0.7946 | 325  | 0.2890          |
| 0.2840        | 0.8557 | 350  | 0.2885          |
| 0.2878        | 0.9169 | 375  | 0.2872          |
| 0.3064        | 0.9780 | 400  | 0.2853          |
| 0.2738        | 1.0391 | 425  | 0.2858          |
| 0.2596        | 1.1002 | 450  | 0.2854          |
| 0.2650        | 1.1614 | 475  | 0.2843          |
| 0.2504        | 1.2225 | 500  | 0.2825          |
| 0.2527        | 1.2836 | 525  | 0.2836          |
| 0.2466        | 1.3447 | 550  | 0.2821          |
| 0.2635        | 1.4059 | 575  | 0.2822          |
| 0.2656        | 1.4670 | 600  | 0.2803          |
| 0.2792        | 1.5281 | 625  | 0.2796          |
| 0.2567        | 1.5892 | 650  | 0.2802          |
| 0.2645        | 1.6504 | 675  | 0.2782          |
| 0.2445        | 1.7115 | 700  | 0.2783          |
| 0.2551        | 1.7726 | 725  | 0.2766          |
| 0.2385        | 1.8337 | 750  | 0.2766          |
| 0.2470        | 1.8949 | 775  | 0.2777          |
| 0.2673        | 1.9560 | 800  | 0.2762          |
| 0.2363        | 2.0171 | 825  | 0.2770          |
| 0.2131        | 2.0782 | 850  | 0.2797          |
| 0.2424        | 2.1394 | 875  | 0.2783          |
| 0.2215        | 2.2005 | 900  | 0.2786          |
| 0.2128        | 2.2616 | 925  | 0.2812          |
| 0.2259        | 2.3227 | 950  | 0.2808          |
| 0.2207        | 2.3839 | 975  | 0.2804          |
| 0.2121        | 2.4450 | 1000 | 0.2784          |
| 0.2199        | 2.5061 | 1025 | 0.2785          |
| 0.2148        | 2.5672 | 1050 | 0.2802          |
| 0.2240        | 2.6284 | 1075 | 0.2790          |
| 0.2206        | 2.6895 | 1100 | 0.2796          |
| 0.2094        | 2.7506 | 1125 | 0.2791          |
| 0.2318        | 2.8117 | 1150 | 0.2779          |
| 0.2352        | 2.8729 | 1175 | 0.2786          |
| 0.2074        | 2.9340 | 1200 | 0.2776          |
| 0.2340        | 2.9951 | 1225 | 0.2768          |
| 0.1952        | 3.0562 | 1250 | 0.2792          |
| 0.2111        | 3.1174 | 1275 | 0.2823          |
| 0.1885        | 3.1785 | 1300 | 0.2826          |
| 0.1967        | 3.2396 | 1325 | 0.2823          |
| 0.2084        | 3.3007 | 1350 | 0.2827          |
| 0.2076        | 3.3619 | 1375 | 0.2829          |
| 0.1974        | 3.4230 | 1400 | 0.2826          |
| 0.2144        | 3.4841 | 1425 | 0.2828          |
| 0.1951        | 3.5452 | 1450 | 0.2833          |
| 0.1995        | 3.6064 | 1475 | 0.2834          |
| 0.2078        | 3.6675 | 1500 | 0.2832          |
| 0.2131        | 3.7286 | 1525 | 0.2831          |
| 0.2136        | 3.7897 | 1550 | 0.2830          |
| 0.2131        | 3.8509 | 1575 | 0.2830          |
| 0.2154        | 3.9120 | 1600 | 0.2830          |
| 0.1992        | 3.9731 | 1625 | 0.2829          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.0.0
- Pytorch 2.10.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2
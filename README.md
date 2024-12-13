# Alternate Preference Optimization for Unlearning Knowledge

We notice that TOFU's Llama2 finetune and retain models have reproducibility issues due to the usage of distributed training. In our experiments, we rely on these model checkpoints and eval logs (in the `data/` folder) in our experiments. For Llama3.2 we train our own models with parameters as mentioned in the paths and configs.

## Sample Commands



## Installation
```
conda create -n tofu python=3.10
conda activate tofu
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

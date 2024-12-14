# Alternate Preference Optimization for Unlearning Knowledge

We find that TOFU's Llama2 finetune and retain models have reproducibility issues due to the usage of distributed training. In our experiments, we rely on these model checkpoints and eval logs (in the `data` folder) in our experiments. For Llama3.2 we train our own models with parameters as mentioned in the paths and configs.

## Sample Commands

### Generate Alternate Dataset

```script
python generate.py dataset_config.dataset_kwargs.name=forget10
python generate.py dataset_config.dataset_kwargs.name=forget05
python generate.py dataset_config.dataset_kwargs.name=forget01
```

## Installation
```script
conda create -n tofu python=3.10
conda activate tofu
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Run AltPO and baselines
```script
python forget.py --config-name=unlearn_llama2.yaml forget_loss=subdpo beta=0.1 retain_wt=1 seed=0 lr=5e-05 num_epochs=2 augment_k=5 batch_size=5
python forget.py --config-name=unlearn_llama2.yaml forget_loss=npo beta=0.05 retain_wt=2 seed=2 lr=2e-05 num_epochs=10 batch_size=5
python forget.py --config-name=unlearn_llama2.yaml forget_loss=idkdpo beta=0.1 retain_wt=1 seed=1 lr=2e-05 num_epochs=10 batch_size=5
```

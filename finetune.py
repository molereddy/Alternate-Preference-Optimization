from data_module import TextDatasetQA, basic_collator
from dataloader import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml

HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')

@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]


    hyperparams = {}
    if cfg.seed is None:
        set_seed(1)
    else:
        set_seed(cfg.seed)
        cfg.save_dir = f"{cfg.save_dir}_{cfg.seed}"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file
    #if master process
    if os.environ.get('LOCAL_RANK') is None:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps)
    # max_steps=5
    print(f"max_steps: {max_steps}")
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            eval_strategy="epoch",
            bf16=True,
            bf16_full_eval=True,
            weight_decay = cfg.weight_decay,
        )
    if cfg.seed is not None:
        training_args.seed = cfg.seed
    
    model_kwargs = {
        'attn_implementation': 'flash_attention_2',
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'cache_dir': HF_HOME
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True # 

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        hyperparams=hyperparams,
        data_collator=basic_collator,
        eval_cfg=cfg.eval,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.evaluate()
    trainer.train()
    trainer.evaluate()
    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()

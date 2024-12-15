from data_module import TextForgetDatasetQA, TextForgetDatasetIDKFullQA, TextForgetDatasetSubFullQA
from dataloader import CustomTrainerForgetting
from data_module import unlearn_collator, unlearn_collator_sub
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
import hydra
import transformers
import pandas as pd
import os, sys, copy, shutil, json
from pathlib import Path
from utils import get_model_identifiers_from_yaml, rearrange_cols
from omegaconf import OmegaConf

HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):

    os.environ["WANDB_DISABLED"] = "true"
    if cfg.seed is None: 
        cfg.seed = 0
    set_seed(cfg.seed)
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    
    
    forget_percentage = int(cfg.split[6:]) # 01, 05 or 10
    cfg.eval.retain_result = os.path.join(model_cfg['retain_evals_path'].format(split=100-forget_percentage), 
                                          'eval_results/ds_size300/eval_log_aggregated.json')

    print("Base cfg before prep is", cfg)
    
    # setup paths
    cfg.model_path = model_cfg["results_path"]
    cfg.eval.model_path = cfg.model_path
    if cfg.save_dir is None:
        cfg.save_dir = f"{cfg.model_path}/{cfg.split}/{cfg.forget_loss}"
        if 'diff' in cfg.forget_loss:
            assert 'alpha' in cfg and cfg.alpha is not None
            cfg.save_dir += f'_alpha_{cfg.alpha}'
        elif 'dpo' in cfg.forget_loss or 'npo' in cfg.forget_loss or 'ppo' in cfg.forget_loss:
            assert 'beta' in cfg and cfg.beta is not None
            cfg.save_dir += f'_beta_{cfg.beta}'
        cfg.save_dir = os.path.join(cfg.save_dir, f"{cfg.lr}_{cfg.seed}")
        cfg.save_dir += f'_ret_{cfg.retain_wt}'
    else:
        assert cfg.model_path in cfg.save_dir
        print("Save path is given to forget.py directly")
    
    shutil.rmtree(cfg.save_dir, ignore_errors=True)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    print("Saving experiment results to: ", os.path.abspath(cfg.save_dir))
    if os.path.exists(cfg.save_dir):
        for item in Path(cfg.save_dir).iterdir():
            if item.is_file() and not (item.suffix in {'.err', '.txt'}):
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        Path(cfg.save_dir).mkdir(parents=True)
    
    # save cfg
    with open(f"{cfg.save_dir}/config.yaml", "w") as file:
        OmegaConf.save(cfg, file)
        
    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 256
    intial_ckpt_ref_needed = (cfg.retain_type == 'KL')
    if 'npo' in cfg.forget_loss or 'ppo' in cfg.forget_loss or 'dpo' in cfg.forget_loss or 'KL' in cfg.forget_loss:
        intial_ckpt_ref_needed = True
        collator = unlearn_collator
    if cfg.forget_loss == "idkdpo": # the idk dpo that the paper found unstable
        torch_format_dataset = TextForgetDatasetIDKFullQA(data_path=cfg.data_path, tokenizer=tokenizer, 
                                                          model_family=cfg.model_family, max_length=max_length, 
                                                          split=cfg.split)
        collator = unlearn_collator_sub
        intial_ckpt_ref_needed = True
    elif cfg.forget_loss in ['subdpo', 'subdiff', 'subnpo', 'subppo_npo', 'subppo']:
        sub_json = os.path.join(cfg.model_path, cfg.split, f'alt{cfg.augment_k}_seed_{cfg.seed}.json')
        torch_format_dataset = TextForgetDatasetSubFullQA(data_path=sub_json, tokenizer=tokenizer, 
                                                          model_family=cfg.model_family, max_length=max_length, 
                                                          split=cfg.split)
        collator = unlearn_collator_sub
    elif cfg.forget_loss == 'sub':
        sub_json = os.path.join(cfg.model_path, cfg.split, f'alt{cfg.augment_k}_seed_{cfg.seed}.json')
        torch_format_dataset = TextForgetDatasetQA(data_path=sub_json, tokenizer=tokenizer,
                                                   model_family=cfg.model_family, max_length=max_length, 
                                                   split=cfg.split, loss_type=cfg.forget_loss)
        collator = unlearn_collator
    else:
        assert cfg.forget_loss in ['grad_diff', 'grad_ascent', 'KL', 'idk', 'npo']
        torch_format_dataset = TextForgetDatasetQA(data_path=cfg.data_path, tokenizer=tokenizer, 
                                                   model_family=cfg.model_family, max_length=max_length, 
                                                   split=cfg.split, loss_type=cfg.forget_loss)
        collator = unlearn_collator
    
    # once dataset collected DPO loss works the same way for all variants
    if 'dpo' in cfg.forget_loss: # idkdpo/subdpo etc. -> dpo
        cfg.forget_loss = cfg.forget_loss[3:]
    
    hyperparams = {
        'retain_wt': cfg.retain_wt, 'retain_type': cfg.retain_type
    }
    if 'diff' in cfg.forget_loss:
        hyperparams['alpha']=cfg.alpha
    elif 'dpo' in cfg.forget_loss or 'npo' in cfg.forget_loss or 'ppo' in cfg.forget_loss:
        hyperparams['beta']=cfg.beta
    
    # count dataset sizes and setup intervals/steps
    batch_size = cfg.batch_size # set to 5
    gradient_accumulation_steps = 35 // cfg.batch_size
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps)
    warmup_steps, logging_steps = max(2, steps_per_epoch), max(1, max_steps//20)
    eval_interval = steps_per_epoch//cfg.augment_k
    eval_strat = "steps" if cfg.eval_while_train else "no"
    if max_steps == 0:
        warmup_steps = 0
        logging_steps = 0
        eval_strat = "no"
    logging_strategy = eval_strat
    save_strategy = "steps" if cfg.save_model and (not cfg.eval_only) else "no"

    training_args = transformers.TrainingArguments(
            seed=cfg.seed,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps, # warming up for an epoch
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_strategy=logging_strategy,
            logging_steps=logging_steps,
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy=save_strategy,
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            weight_decay = cfg.weight_decay,
            report_to='tensorboard',
            eval_steps = eval_interval,
            bf16 = True,
            bf16_full_eval = False,
            eval_strategy = eval_strat,
        )
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}", 
          f"max_steps: {max_steps}", f"steps_per_epoch: {steps_per_epoch}",
          f"warmup_steps: {warmup_steps}")

    # load model(s)
    model_kwargs = {
        'attn_implementation': 'flash_attention_2',
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'cache_dir': HF_HOME
    }
    load_from = model_cfg["ft_model_path"]
    model = AutoModelForCausalLM.from_pretrained(load_from, **model_kwargs)
    model.generation_config.do_sample = True # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    oracle_model = None
    if intial_ckpt_ref_needed:
        oracle_model = copy.deepcopy(model).to('cuda')
        oracle_model.eval()

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=collator,
        hyperparams=hyperparams,
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        eval_cfg=cfg.eval,
    )
    if cfg.start_with_eval:
        trainer.evaluate()
    if not cfg.eval_only:
        trainer.train()
    if cfg.save_model:
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        
    # find results.csv in cfg.save_dir

if __name__ == "__main__":
    main()
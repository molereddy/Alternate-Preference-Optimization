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
from utils import get_model_identifiers_from_yaml, get_model_utility_v2, rearrange_cols
from omegaconf import OmegaConf

HF_HOME = os.getenv('HF_HOME', '~/.cache/huggingface')

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):

    os.environ["WANDB_DISABLED"] = "true"
    if cfg.seed is None: 
        cfg.seed = 0
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    
    
    forget_percentage = int(cfg.split[6:]) # 01, 05 or 10
    cfg.eval.retain_result = os.path.join(model_cfg['retain_evals_path'].format(split=100-forget_percentage), 
                                          'eval_results/ds_size300/eval_log_aggregated.json')
    
    set_seed(cfg.seed)
    
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
    
    shutil.rmtree(cfg.save_dir, ignore_errors=True)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    
    # save cfg
    with open(f"{cfg.save_dir}/config.yaml", "w") as file:
        OmegaConf.save(cfg, file)
        
    # setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.exists(cfg.save_dir):
        # print(f"Method directory already exists:{os.path.abspath(cfg.save_dir)}")
        if not cfg.overwrite_dir:
            exit()
    if os.path.exists(os.path.join(cfg.save_dir, 'trajectory.png')):
        print("Results already exist in directory, skipping experiment")
        sys.exit()

    max_length = 256
    collator=unlearn_collator
    
    intial_ckpt_ref_needed = (cfg.retain_type == 'KL')
    if 'npo' in cfg.forget_loss or 'ppo' in cfg.forget_loss or 'dpo' in cfg.forget_loss or 'KL' in cfg.forget_loss:
        intial_ckpt_ref_needed=True
    if cfg.forget_loss == "idkdpo": # the idk dpo that the paper found unstable
        torch_format_dataset = TextForgetDatasetIDKFullQA(data_path=cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split)
        collator=unlearn_collator_sub
        intial_ckpt_ref_needed = True
    elif cfg.forget_loss in ['subdpo', 'subdpop', 'subdiff', 'subnpo', 'subppo_npo', 'subppo']:
        sub_json = os.path.join(cfg.model_path, cfg.split, f'alt{cfg.augment_k}_seed_{cfg.seed}.json')
        # sub_json = os.path.join(cfg.model_path, cfg.split, f'sample_alt.json')
        torch_format_dataset = TextForgetDatasetSubFullQA(data_path=sub_json, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split)
        collator=unlearn_collator_sub
    elif cfg.forget_loss == 'sub':
        sub_json = os.path.join(cfg.model_path, cfg.split, f'alt{cfg.augment_k}_seed_{cfg.seed}.json')
        # sub_json = os.path.join(cfg.model_path, cfg.split, f'sample_alt.json')
        torch_format_dataset = TextForgetDatasetQA(data_path=sub_json, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss)
    else:
        assert cfg.forget_loss in ['base', 'grad_diff', 'grad_ascent', 'KL', 'idk', 'npo']
        torch_format_dataset = TextForgetDatasetQA(data_path=cfg.data_path, tokenizer=tokenizer, model_family=cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss)
    
    # simplify DPO: subdpo -> dpo, subdpop -> dpop 
    if cfg.forget_loss=='idkdpo' or 'dpo' in cfg.forget_loss: # once dataset is collected
        cfg.forget_loss = cfg.forget_loss[3:] #  dataset type doesn't matter, only loss type does
    
    hyperparams = {
        'retain_wt': cfg.retain_wt, 'retain_type': cfg.retain_type, 'weights_scheduler': cfg.weights_scheduler
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
    warmup_steps = max(2, steps_per_epoch) if max_steps != 0 else 0 # warming up for an epoch
    eval_interval = steps_per_epoch
    eval_interval //= cfg.augment_k

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=cfg.lr,
            logging_steps=max(2,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            weight_decay = cfg.weight_decay,
            report_to='tensorboard',
            eval_steps = eval_interval,
            evaluation_strategy = "steps" if cfg.eval_while_train else "no",
        )
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}", 
          f"max_steps: {max_steps}", f"steps_per_epoch: {steps_per_epoch}",
          f"warmup_steps: {warmup_steps}")
    if cfg.seed is not None:
        training_args.seed = cfg.seed
    
    # load model(s)
    model_kwargs = {
        'attn_implementation': 'flash_attention_2',
        'torch_dtype': torch.bfloat16,
        'trust_remote_code': True,
        'cache_dir': HF_HOME
    }
    load_from = model_cfg["ft_model_path"]
    model = AutoModelForCausalLM.from_pretrained(load_from, **model_kwargs)
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    oracle_model = None
    if intial_ckpt_ref_needed:
        oracle_model = copy.deepcopy(model).to('cuda')
        oracle_model.eval()
    training_args.bf16 = True

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
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if cfg.start_with_eval is None or cfg.start_with_eval:
        trainer.evaluate()
    if not cfg.eval_only and cfg.forget_loss != "base":
        print("Train started")
        trainer.train()
    if cfg.save_model:
        trainer.save_model()
    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
    
    # results_csv = os.path.join(cfg.save_dir, 'results.csv')
    
    # results_df = pd.read_csv(results_csv)
    # new_mu_values = []
    
    # for ckpt in os.listdir(cfg.save_dir):
    #     if not ckpt.startswith("checkpoint"):
    #         continue
    #     ckpt_path = os.path.join(cfg.save_dir, ckpt)
    #     if not os.path.isfile(os.path.join(ckpt_path, 'eval_log_forget.json')):
    #         continue
    #     step = int(ckpt.split('-')[1])
    #     gibberish_config = OmegaConf.load(Path('config/eval_gibberish.yaml'))
    #     add_gibberish_evals(ckpt_path, gibberish_config)
    #     agg_eval_log_fname = os.path.join(ckpt_path, "eval_log_aggregated.json")
    #     with open(agg_eval_log_fname, 'r') as f:
    #         aggregated_eval_logs = json.load(f)
    #     new_ckpt_values = get_model_utility_v2(aggregated_eval_logs, json.load(open(cfg.eval.retain_result, 'r')))
    #     new_ckpt_values['step']=step
    #     new_mu_values.append(new_ckpt_values)
    
    # assert len(new_mu_values) == len(results_df)
    # new_mu_values = pd.DataFrame(new_mu_values)
    # new_mu_values.sort_values(by='step', inplace=True)
    
    # columns_to_drop = [col for col in new_mu_values.columns if col in results_df.columns and col != 'step']
    # new_mu_values = new_mu_values.drop(columns=columns_to_drop)
    # results_df = results_df.merge(new_mu_values, on='step', how='inner')
    # assert not results_df.isnull().values.any(), "Null values in merged df"
    # results_df = rearrange_cols(results_df)
    # results_df.to_csv(results_csv, index=False)
    
    # plot_trajectory(results_csv)
    # plot_histograms(cfg.save_dir)
    

if __name__ == "__main__":
    main()
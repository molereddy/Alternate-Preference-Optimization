import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import os, json, glob, csv, yaml
import numpy as np
import pandas as pd
from tqdm import tqdm       
from omegaconf import OmegaConf
from scipy.stats import sem, hmean, ks_2samp


def gibberish_evals(dir_path, retain_logs):
    
    model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")
    
    unlearn_model_logs_path = os.path.join(dir_path, "eval_log_aggregated.json")
    unlearn_logs = json.load(open(unlearn_model_logs_path, 'r'))["eval_log_forget.json"]
    
    unlearn_texts = [v[1] for k, v in unlearn_logs['generated_text'].items()]
    retain_texts = [v[1] for k, v in retain_logs['generated_text'].items()]
    
    bsz = 80
    unlearn_scores = []
    for idx in tqdm(range(0, len(unlearn_texts), bsz)):
        batch = unlearn_texts[idx:idx+bsz]
        tokenized_sentences = tokenizer(batch, max_length=256,truncation=True, 
                                        padding=True,return_tensors="pt", 
                                        return_attention_mask=True)
        tokenized_sentences = {k: v.to("cuda") for k, v in tokenized_sentences.items()}
        with torch.no_grad():
            outputs = model(**tokenized_sentences)
        probs = F.softmax(outputs.logits, dim=-1)[:, 0].cpu()
        probs = probs.to(dtype=torch.float32).numpy().tolist()
        unlearn_scores.extend(probs)
    unlearn_scores = np.array(unlearn_scores)
    retain_scores = []
    for idx in tqdm(range(0, len(retain_texts), bsz)):
        batch = retain_texts[idx:idx+bsz]
        tokenized_sentences = tokenizer(batch, max_length=256,truncation=True, 
                                        padding=True,return_tensors="pt", 
                                        return_attention_mask=True)
        tokenized_sentences = {k: v.to("cuda") for k, v in tokenized_sentences.items()}
        with torch.no_grad():
            outputs = model(**tokenized_sentences)
        probs = F.softmax(outputs.logits, dim=-1)[:, 0].cpu()
        probs = probs.to(dtype=torch.float32).numpy().tolist()
        retain_scores.extend(probs)
    retain_scores = np.array(retain_scores)
    TC = round(np.mean(unlearn_scores), 3)
    CI = ks_2samp(unlearn_scores, retain_scores).pvalue
    return TC, CI


def main():
    exp_path = 'paper_models/tofu_ft_llama2-7b/forget10/subdpo_beta_0.1/5e-05_1_ret_1'
    results_df = pd.read_csv(os.path.join(exp_path, 'results.csv'))
    new_mu_values = []
    config_yaml = os.path.join(exp_path, 'config.yaml')
    exp_config = yaml.safe_load(open(config_yaml, 'r'))
    retain_logs = json.load(open(exp_config['eval']['retain_result'], 'r'))["eval_log_forget.json"]
    
    for checkpoint in os.listdir(exp_path):
        if not checkpoint.startswith("checkpoint"):
            continue
        checkpoint_path = os.path.join(exp_path, checkpoint)
        print(checkpoint_path)
        if not os.path.isfile(os.path.join(checkpoint_path, 'eval_log_forget.json')): continue
        print(gibberish_evals(checkpoint_path, retain_logs))
    
if __name__ == "__main__":
    main()

import os, json, yaml, copy
import copy
import torch
import numpy as np
from scipy.stats import hmean, ks_2samp
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


column_order = ['step', 'Model Utility', 'Forget Quality', 'TC', 'CI',
                'Forget Probability', 'Forget Cleanness',
                'Forget Paraphrase', 'Forget Perturbed', 
                'Forget Truth Ratio', 'Real Authors ROUGE', 
                'Real Authors Probability', 'Real Authors Truth Ratio', 
                'Real World ROUGE', 'Real World Probability', 
                'Real World Truth Ratio', 'Retain ROUGE', 
                'Retain Probability', 'Retain Truth Ratio', 
                'Forget ROUGE', 'KS Test Forget']

def get_model_identifiers_from_yaml(model_family):
    model_configs  = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def get_model_utility(eval_result_dict):
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_log' in k: # tofu datasets
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge
        data_indices = list(eval_result_dict[k]['avg_paraphrased_loss'].keys())
        avg_paraphrase_np_values = []
        avg_perturbed_np_values = []
        for data_idx in data_indices:
            avg_paraphrase_np_values.append(eval_result_dict[k]['avg_paraphrased_loss'][data_idx])
            avg_perturbed_np_values.append(eval_result_dict[k]['average_perturb_loss'][data_idx])

        avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))
        avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        r_truth =  np.exp(-1* (avg_perturbed_np_values - avg_paraphrase_np_values)) # R_truth
        if eval_task_dict[k]=='Forget':
            output_result[f'Forget Paraphrase'] = np.mean(np.exp(-1*avg_paraphrase_np_values))
            output_result[f'Forget Perturbed'] = np.mean(np.exp(-1*avg_perturbed_np_values))
        truth_ratio = np.mean(np.minimum(r_truth, 1/r_truth)) if 'forget' in k else np.mean(np.maximum(0, 1 - r_truth))  
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = truth_ratio

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    
    return output_result


def get_gibberish_evals(dir_path, retain_logs):
    
    print("Evaluating gibberish at path:", dir_path)
    model = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457", torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")
    
    unlearn_model_logs_path = os.path.join(dir_path, "eval_log_aggregated.json")
    unlearn_logs = json.load(open(unlearn_model_logs_path, 'r'))["eval_log_forget.json"]
    unlearn_texts = [v[1] for k, v in unlearn_logs['generated_text'].items()]
    retain_logs = retain_logs["eval_log_forget.json"]
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
    
    return {'TC': TC, 'CI': CI}


def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    results = {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}
    
    return results


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset
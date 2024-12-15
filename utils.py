import yaml
import copy
import torch
import numpy as np
from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted
column_order = ['step', 'Model Utility', 'logFQ', 
        'Forget Quality', 'Forget Probability', 'Forget Cleanness',
        'Forget Paraphrase', 'Forget Perturbed', 'Forget Truth Ratio', 
        'Real Authors ROUGE', 'Real Authors Probability', 'Real Authors Truth Ratio', 
        'Real World ROUGE', 'Real World Probability', 'Real World Truth Ratio', 
        'Retain ROUGE', 'Retain Probability', 'Retain Truth Ratio', 
        'Forget ROUGE', 'KS Test Forget']

def rearrange_cols(df):
    cols_ = [col for col in column_order if col in df.columns]
    return df[cols_]

def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
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
        if 'eval_log' in k:
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

        # if 'avg_paraphrased_loss' not in eval_result_dict[k]: ## ADD BACK FOR FINETUNE
        #     continue
        # getting Truth Ratio
        data_indices = list(eval_result_dict[k]['avg_paraphrased_loss'].keys())
        # #------------
        # # tofu aggregate stat version
        # avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))
        # avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
        # #------------
        # group avg_paraphrased_loss and average_perturb_loss by data_indices
        avg_paraphrase_np_values = []
        avg_perturbed_np_values = []
        for data_idx in data_indices:
            avg_paraphrase_np_values.append(eval_result_dict[k]['avg_paraphrased_loss'][data_idx])
            avg_perturbed_np_values.append(eval_result_dict[k]['average_perturb_loss'][data_idx])

        avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values())) # loss
        avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values())) # loss
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1) # loss

        r_truth =  np.exp(-1* (avg_perturbed_np_values - avg_paraphrase_np_values)) # R_truth
        if eval_task_dict[k]=='Forget':
            output_result[f'Forget Paraphrase'] = np.mean(np.exp(-1*avg_paraphrase_np_values))
            output_result[f'Forget Perturbed'] = np.mean(np.exp(-1*avg_perturbed_np_values))
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        truth_ratio = np.mean(np.minimum(r_truth, 1/r_truth)) if 'forget' in k else np.mean(np.maximum(0, 1 - r_truth))
            
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = truth_ratio


    mu_cand_keys = ['Real Authors ROUGE','Real Authors Probability', 'Real Authors Truth Ratio',
                    'Real World ROUGE', 'Real World Truth Ratio', 'Real World Probability',
                    'Retain Truth Ratio', 'Retain Probability', 'Retain ROUGE']
    mu_cand_vals = []
    candidate_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    ignore = []
    for k, v in output_result.items():
        try:
            _ = round(v, 3)
        except Exception as e:
            ignore.append(k)
            continue
        if k in mu_cand_keys:
            mu_cand_vals.append(v)
    for k in ignore:
        _ = output_result.pop(k)
    if len(mu_cand_vals)==len(candidate_weights):
        output_result['Model Utility'] = round(hmean(mu_cand_vals, weights=candidate_weights), 3)

    return output_result

def get_fq_v3(eval_result_dict, retain_logs):
    forget_dict = eval_result_dict['eval_log_forget.json']
    unlearn_ppl = np.array(list(forget_dict['llama3_ppl'].values()))
    retain_ppl = np.array(list(retain_logs['eval_log_forget.json']['llama3_ppl'].values()))
    
    return {'FQ V3':np.log10(ks_2samp(unlearn_ppl, retain_ppl).pvalue)}

def get_model_utility_v2(eval_result_dict, retain_logs):
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
        if 'eval_log' in k:
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

        # getting Truth Ratio
        data_indices = list(eval_result_dict[k]['avg_paraphrased_loss'].keys())
        avg_paraphrase_np_values = []
        avg_perturbed_np_values = []
        for data_idx in data_indices:
            avg_paraphrase_np_values.append(eval_result_dict[k]['avg_paraphrased_loss'][data_idx])
            avg_perturbed_np_values.append(eval_result_dict[k]['average_perturb_loss'][data_idx])

        avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values())) # loss
        avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values())) # loss
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1) # loss

        r_truth =  np.exp(-1* (avg_perturbed_np_values - avg_paraphrase_np_values)) # R_truth
        truth_ratio = np.mean(np.minimum(r_truth, 1/r_truth)) if 'forget' in k else np.mean(np.maximum(0, 1 - r_truth))
            
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = truth_ratio

    # cleanness
    forget_dict = eval_result_dict['eval_log_forget.json']
    unlearn_cleanness = np.array(list(forget_dict['cleanness'].values()))
    retain_cleanness = np.array(list(retain_logs['eval_log_forget.json']['cleanness'].values()))
    avg_clean_score = np.mean(unlearn_cleanness)
    output_result[f'Forget Cleanness'] = round(avg_clean_score, 3)
    
    mu_cand_keys = ['Real Authors ROUGE','Real Authors Probability','Real World ROUGE',
                    'Real World Probability','Retain Truth Ratio', 'Retain Probability', 
                    'Retain ROUGE']
    mu_cand_vals = []
    candidate_weights = np.array([0.25, 0.25, 0.25, 0.25, 1/3, 1/3, 1/3])
    for k, v in output_result.items():
        if k in mu_cand_keys:
            mu_cand_vals.append(v)
    test_res = ks_2samp(unlearn_cleanness, retain_cleanness)
    output_result['FQ V2'] = np.log10(test_res.pvalue)
    return output_result

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
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}


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
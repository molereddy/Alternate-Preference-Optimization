import os
import csv
import json
import torch
import numpy as np
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
from pathlib import Path

from evaluate_util import get_dataloader, get_all_evals
from utils import get_forget_quality, get_model_utility, get_gibberish_evals
from utils import get_batch_loss, column_order

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        self.eval_cfg = kwargs.pop('eval_cfg')
        super(CustomTrainer, self).__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        
        hyperparams = kwargs.pop('hyperparams', {})
        self.alpha = hyperparams.get('alpha')
        self.beta = hyperparams.get('beta')
        self.retain_wt = hyperparams.get('retain_wt')
        self.retain_type = hyperparams.get('retain_type')

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        
        def get_retain_loss(retain_inputs):
            loss = torch.tensor(0.0).to(self.args.device)
            if self.retain_type == 'NLL':
                loss = NLL(retain_inputs)[0]
            elif self.retain_type == 'KL':
                loss = KL(retain_inputs)[0]
                print('KL loss:', loss)
            else:
                raise NotImplementedError
            return self.retain_wt * loss
        
        def NLL(inputs):
            input_ids, labels, attention_mask = inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            return outputs.loss, outputs
        
        def batchNLL(model, inputs):
            input_ids, labels, attention_mask = inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            return get_batch_loss(outputs.logits, labels), outputs
            
        def KL(inputs):
            input_ids, labels, attention_mask = inputs
            with torch.no_grad():
                ref_outputs = self.oracle_model(input_ids, labels=labels, attention_mask=attention_mask)
            
            ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
            ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])
            
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            current_probs = F.log_softmax(outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, outputs.logits.shape[-1])

            #minimum KL divergence
            return nn.functional.kl_div(current_probs, ref_probs, reduction='batchmean', log_target=True), outputs
            
        def PO(inputs, beta=0.1, lose=True):
            curr_loss, outputs = batchNLL(model, inputs) 
            with torch.no_grad():
                ref_loss, _ = batchNLL(self.oracle_model, inputs)
            log_ratios = -(curr_loss - ref_loss) # loss is neg of log probs
            if not lose: log_ratios *= -1 # the ppo version of NPO
            loss = -2/beta * F.logsigmoid(-beta * log_ratios).mean()
            return loss, outputs

        def DPO(win_inputs, lose_inputs, beta=0.1):
            lose_loss_curr, outputs = batchNLL(model, lose_inputs)
            win_loss_curr, _ = batchNLL(model, win_inputs)
            with torch.no_grad():
                lose_loss_ref, _ = batchNLL(self.oracle_model, lose_inputs)
                win_loss_ref, _ = batchNLL(self.oracle_model, win_inputs)
            log_ratios_win = -(win_loss_curr - win_loss_ref)
            log_ratios_lose = -(lose_loss_curr - lose_loss_ref)
            loss = -2/beta * F.logsigmoid(beta * (log_ratios_win-log_ratios_lose)).mean()
            return loss, outputs
        
        loss = torch.tensor(0.0).to(self.args.device)
        if self.loss_type == "grad_ascent":
            forget_inputs, _ = inputs
            
            nll_forget, outputs = NLL(forget_inputs)
            loss -= nll_forget

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
    
            nll_forget, outputs = NLL(forget_inputs)
            loss -= nll_forget
            retain_loss = get_retain_loss(retain_inputs=retain_inputs)
            loss += retain_loss
        
        elif self.loss_type == "KL":
            # same as grad_diff with retain_type as KL
            forget_inputs, retain_inputs = inputs
            nll_forget, outputs = NLL(forget_inputs)
            loss -= nll_forget
            kl_retain = self.retain_wt * KL(retain_inputs)[0]
            loss += kl_retain

        elif self.loss_type in ["idk", "sub"]:
            sub_inputs, retain_inputs = inputs
            sub_input_ids, sub_labels, sub_attention_mask = sub_inputs

            NLL_sub = model(sub_input_ids, labels=sub_labels, attention_mask=sub_attention_mask).loss
            retain_loss = get_retain_loss(retain_inputs=retain_inputs)
            
            loss += NLL_sub
            loss += retain_loss
        
        elif self.loss_type == "subdiff":
            assert self.alpha is not None
            sub_inputs, forget_inputs, retain_inputs = inputs

            nll_forget, outputs = NLL(forget_inputs)
            nll_sub = NLL(sub_inputs)[0]

            loss -= self.alpha * nll_forget
            loss += nll_sub
            retain_loss = get_retain_loss(retain_inputs=retain_inputs)
            loss += retain_loss
        
        elif self.loss_type == "dpo":
            assert self.beta is not None
            sub_inputs, forget_inputs, retain_inputs = inputs # sub/idk, forget and retain
            dpo_sub_forget, outputs = DPO(sub_inputs, forget_inputs, self.beta) 
            loss += dpo_sub_forget
            retain_loss = get_retain_loss(retain_inputs=retain_inputs)
            loss += retain_loss  
        
        elif 'npo' in self.loss_type:
            assert self.beta is not None
            if self.loss_type == 'npo':
                forget_inputs, retain_inputs = inputs
            else:
                sub_inputs, forget_inputs, retain_inputs = inputs # sub/idk, forget and retain
            po_forget, outputs = PO(forget_inputs, self.beta, lose=True)
            loss += po_forget
            if self.loss_type == 'subnpo':
                nll_sub = NLL(sub_inputs)[0]
                loss += nll_sub
            elif self.loss_type == 'subppo_npo':
                po_sub = PO(sub_inputs, self.beta, lose=False)[0]
                loss += po_sub 
            retain_loss = get_retain_loss(retain_inputs=retain_inputs)
            loss += retain_loss  
        
        elif 'subppo' in self.loss_type:
            sub_inputs, forget_inputs, retain_inputs = inputs # sub/idk, forget and retain
            po_sub = PO(sub_inputs, self.beta, lose=False)[0]
            loss += po_sub

            retain_loss = get_retain_loss(retain_inputs=retain_inputs)
            loss += retain_loss
        
        else:
            assert False
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg
        print('---'*40)
        print(model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}") if self.loss_type != "base" else eval_cfg.save_dir
        print("Saving eval in", curr_save_dir)
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        
        aggregated_eval_logs = {}
        model.eval()
        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                normalize_gt = False 
                if 'eval_log' not in eval_task:
                    normalize_gt = True
                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)
                aggregated_eval_logs[f'{eval_task}.json'] = eval_logs
                
                with open(save_filename, "w") as f: # dump logs to eval task
                    json.dump(eval_logs, f, indent=4)
        
        print('---'*40)
        agg_evals_dump_path = os.path.join(curr_save_dir, "eval_log_aggregated.json")
        with open(agg_evals_dump_path, 'w') as f:
            json.dump(aggregated_eval_logs, f, indent=4)
        aggregated_eval_logs = json.load(open(agg_evals_dump_path, 'r'))
        
        assert eval_cfg.retain_result
        retain_eval_logs = json.load(open(eval_cfg.retain_result, 'r'))
        gibberish_scores = get_gibberish_evals(curr_save_dir, retain_eval_logs)
        forget_quality = get_forget_quality(aggregated_eval_logs, retain_eval_logs)
        model_utility = get_model_utility(aggregated_eval_logs)
        stat = {**model_utility, **forget_quality, **gibberish_scores}
        stat['step'] = curr_step
        stat_copy = {}
        # round up and reorder results to use in paper
        for k in column_order:
            if k in ['Forget Quality', 'CI']:
                stat_copy[k] = "{:.3e}".format(stat[k])
            elif k != 'step':
                stat_copy[k] = round(stat[k], 3)
            else:
                stat_copy[k] = stat[k]
        csv_path = os.path.join(eval_cfg.save_dir, 'results.csv')
        mode = 'a' if os.path.exists(csv_path) else 'w'
        with open(csv_path, mode) as f:
            w = csv.DictWriter(f, fieldnames=column_order)
            if mode == 'w':
                w.writeheader()
            w.writerow(stat_copy)
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    assert pad_length > 0
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


class TextForgetDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
        super(TextForgetDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = None
        if data_path == "locuslab/TOFU":
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
            self.mode = "idk" if loss_type == "idk" else "base"
        else:
            self.forget_data = datasets.load_dataset('json', data_files=data_path)["train"]
            self.mode = "sub"
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset("locuslab/TOFU", retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type

        if self.loss_type == "idk":
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        
        # forget
        question = self.forget_data[idx]['question']
        answer = None
        if self.mode == "base":
            answer = self.forget_data[idx]['answer']
        elif self.mode == "sub":
            answer = self.forget_data[idx]['sub_answer']
        elif self.mode == "idk": #get a random answer position from idk
            pos = torch.randint(0, len(self.idk), (1,)).item()
            answer = self.idk[pos].strip()
        else:
            assert False
        converted_forget = convert_raw_data_to_model_format(self.tokenizer, self.max_length, 
                                                            question, answer, self.model_configs)
        rets.append(converted_forget)
        
        # random retain example
        idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        converted_retain = convert_raw_data_to_model_format(self.tokenizer, self.max_length, 
                                                            self.retain_data[idx]['question'], 
                                                            self.retain_data[idx]['answer'], 
                                                            self.model_configs)
        rets.append(converted_retain)
        return rets

class TextForgetDatasetIDKFullQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10", ):
        super(TextForgetDatasetIDKFullQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset(data_path, split)["train"]
        self.idontknowfile = "data/idontknow.jsonl"
        self.idk = open(self.idontknowfile, "r").readlines()
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []

        for data_type in ["idk", "forget", "retain"]:
            data = self.forget_data if data_type != "retain" else self.retain_data
            idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
            
            question = data[idx]['question']
            
            if data_type != "idk":
                answer = data[idx]['answer']
            else:
                #get a random position from idk
                rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                answer = self.idk[rand_pos].strip()

            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            rets.append(converted_data)
        return rets

class TextForgetDatasetSubFullQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = "forget10"):
        super(TextForgetDatasetSubFullQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.forget_data = datasets.load_dataset('json', data_files=data_path)["train"]
        retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
        self.retain_data = datasets.load_dataset("locuslab/TOFU", retain_split)["train"]
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        rets = []
        # forget_sub
        converted_forget_sub = convert_raw_data_to_model_format(self.tokenizer, self.max_length, 
                                                            self.forget_data[idx]['question'], 
                                                            self.forget_data[idx]['sub_answer'], 
                                                            self.model_configs)
        rets.append(converted_forget_sub)
        # forget original
        converted_forget = convert_raw_data_to_model_format(self.tokenizer, self.max_length, 
                                                            self.forget_data[idx]['question'], 
                                                            self.forget_data[idx]['answer'], 
                                                            self.model_configs)
        rets.append(converted_forget)
        # random retain example
        idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        converted_retain = convert_raw_data_to_model_format(self.tokenizer, self.max_length, 
                                                            self.retain_data[idx]['question'], 
                                                            self.retain_data[idx]['answer'], 
                                                            self.model_configs)
        rets.append(converted_retain)
        return rets

      
class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split = None, question_key='question', answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.data = datasets.load_dataset(data_path, split)["train"]
        self.data = add_dataset_index(self.data)
        
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])


        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)

class TextDatasetGenQA(Dataset):
    def __init__(self, gen_texts_dict, tokenizer, model_cfg, max_length=512):
        super(TextDatasetGenQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = model_cfg
        
        self.data = {int(k):[self.remove_existing_tags(v[0]),
                             self.remove_existing_tags(v[1])] for k, v in gen_texts_dict.items()}
        self.data = np.array([self.data[key] for key in sorted(self.data.keys())])
    
    def remove_existing_tags(self, sample_text):
        start_tags = ["[INST] ", "Question: ", "Answer: "]
        end_tags = [" [/INST]", "\n"]
        for tag in start_tags: 
            if sample_text.startswith(tag): 
                sample_text = sample_text[len(tag):]
        for tag in end_tags: 
            if sample_text.endswith(tag): 
                sample_text = sample_text[:-len(tag)]
        return sample_text
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx, 0]
        answer = self.data[idx, 1]
        indices = idx
        converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)

        return converted_data[0].squeeze(),converted_data[1].squeeze(),\
                converted_data[2].squeeze(), torch.tensor(indices)

# collators covert dataset from lists to torch tensor formats
# these collators are for eval and finetune stage3_param_persistence_threshold
# forget collators are in dataloader.py
def basic_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)

def unlearn_collator(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

def unlearn_collator_sub(samples):
    sub_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
    rets = []
    for data_type in ["sub", "forget", "retain"]:
        data = (sub_samples if data_type == "sub" else forget_samples if data_type == "forget" else retain_samples)
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets

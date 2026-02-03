# !/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import json


class MyDataset(Dataset):
    def __init__(self, path):
        self.dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                j = json.loads(line)
                self.dataset.append([j['input'],j['target']])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class BaseCollator:
    def __init__(self, tokenizer, input_max_length, output_max_length, model_type):
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.outupt_max_length = output_max_length
        self.pad_id = tokenizer.pad_token_id
        self.model_type = model_type

    def __call__(self, features, return_tensors=None):
        raise NotImplementedError

    def build_glm3_input(self, input, label):
        input_ids = self.tokenizer.build_single_message('user', "", input)
        if len(input_ids) > self.input_max_length - 5:
            input_ids = input_ids[:self.input_max_length - 5]
        input_ids = self.tokenizer.get_prefix_tokens() + input_ids + [
            self.tokenizer.get_command("<|assistant|>")] + self.tokenizer.encode('\n', add_special_tokens=False)
        label_ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(label_ids) > self.outupt_max_length - 1:
            label_ids = label_ids[:self.outupt_max_length - 1]
        label_ids = label_ids + [self.tokenizer.eos_token_id]
        return input_ids, label_ids

    def build_qwen_input(self, input, label):
        message = [{'role':'user', 'content':input}]
        input_ids = self.tokenizer.apply_chat_template(message, truncation=True, max_length = self.input_max_length, add_generation_prompt=True)
        output_ids = self.tokenizer.encode(label, add_special_tokens=False)
        if len(output_ids) > self.outupt_max_length - 1:
            output_ids = output_ids[:self.outupt_max_length - 1]
        output_ids = output_ids + [self.tokenizer.eos_token_id]
        return input_ids, output_ids

class TrainCollator(BaseCollator):
    def __call__(self, features):
        bsz = len(features)
        input_ids = torch.full((bsz, self.input_max_length + self.outupt_max_length), self.pad_id).long()
        labels = torch.full((bsz, self.input_max_length + self.outupt_max_length), -100).long()
        attention_mask = torch.zeros((bsz, self.input_max_length + self.outupt_max_length)).long()
        
        for i, item in enumerate(features):
            input_text, label_text = item
            if self.model_type == 'glm3':
                input_ids_one, label_ids_one = self.build_glm3_input(input_text, label_text)
            elif self.model_type == 'qwen':
                input_ids_one, label_ids_one = self.build_qwen_input(input_text, label_text)
            else:
                raise ValueError('model_type incorrect')
            
            input_length = len(input_ids_one)
            label_length = len(label_ids_one)
            input_ids[i,:input_length+label_length] = torch.LongTensor(input_ids_one+label_ids_one)
            labels[i,input_length:input_length+label_length] = torch.LongTensor(label_ids_one)
            attention_mask[i, :input_length+label_length] = 1
            
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class EvalCollator(BaseCollator):
    def __call__(self, features):
        bsz = len(features)
        input_ids = torch.full((bsz, self.input_max_length), self.pad_id).long()
        labels = torch.full((bsz, self.outupt_max_length), self.pad_id).long()
        attention_mask = torch.zeros((bsz, self.input_max_length)).long()
        
        for i, item in enumerate(features):
            input_text, label_text = item
            if self.model_type == 'glm3':
                input_ids_one, label_ids_one = self.build_glm3_input(input_text, label_text)
            elif self.model_type == 'qwen':
                input_ids_one, label_ids_one = self.build_qwen_input(input_text, label_text)
            else:
                raise ValueError('model_type incorrect')
            
            input_ids[i,-len(input_ids_one):] = torch.LongTensor(input_ids_one)
            labels[i,:len(label_ids_one)] = torch.LongTensor(label_ids_one)
            attention_mask[i, -len(input_ids_one):] = 1
            
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


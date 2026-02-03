# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import transformers
from peft import get_peft_model, LoraConfig, TaskType, PromptEncoderConfig, PeftModel
import yaml
from data import MyDataset
from model import CustomTrainer
from utils import AutoSaveLogCallback


def get_model(model_path:str, peft_config:dict, do_train:bool=True, peft_model_path:str=None):
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True, device_map="cuda", torch_dtype='auto')
    if do_train:
        if peft_config['peft_type'] == 'ptuning':
            config = peft_config['ptuning_config']
            config.update({'task_type':TaskType.CAUSAL_LM})
            config = PromptEncoderConfig(**config)
        elif peft_config['peft_type'] == 'lora':
            config = peft_config['lora_config']
            config.update({'task_type': TaskType.CAUSAL_LM})
            config = LoraConfig(**config)
        else:
            raise ValueError
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        assert peft_model_path is not None
        model = PeftModel.from_pretrained(model, peft_model_path)
    return model

def get_tokenizer(model_path:str):
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    return tokenizer

def main(config):
    train_config = config['train']
    peft_config = config['peft']
    
    transformers.set_seed(train_config['seed'])
    
    model = get_model(train_config['model_path'], peft_config, train_config['do_train'], train_config['peft_model_path'])
    tokenizer = get_tokenizer(train_config['model_path'])
    print(f"model.device: {model.device}")
    print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")

    train_dataset = MyDataset(train_config['train_data_path'])
    eval_dataset = MyDataset(train_config['eval_data_path'])
    test_dataset = MyDataset(train_config['test_data_path'])

    os.makedirs(train_config['output_dir'], exist_ok=True)

    if train_config['do_train']:
        training_args = TrainingArguments(
            output_dir=train_config['output_dir'],
            num_train_epochs=train_config['epochs'],
            per_device_train_batch_size=train_config['train_batch_size'],
            per_device_eval_batch_size=train_config['eval_batch_size'],
            gradient_accumulation_steps=4,
            learning_rate=train_config['lr'],
            save_steps=500,
            save_total_limit=3,
            logging_steps=10,
            eval_strategy="steps" if train_config['eval_steps'] else "epoch",
            eval_steps=train_config['eval_steps'] if train_config['eval_steps'] else 500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_main",  # 使用自定义的主要指标
            greater_is_better=True,  # rouge-l越高越好
            dataloader_drop_last=False,
            remove_unused_columns=False,
        )

        auto_save_callback = AutoSaveLogCallback(
            output_dir=train_config['output_dir']
        )

        trainer = CustomTrainer(
            tokenizer=tokenizer,
            input_max_length=train_config['input_max_length'],
            output_max_length=train_config['output_max_length'],
            enable_save_model=train_config['do_train'],
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_type=train_config['model_type'],
            callbacks=[auto_save_callback]
        )
        
        trainer.train()
        
        print("训练完成！")
        
    else:
        training_args = TrainingArguments(
            output_dir=train_config['output_dir'],
            per_device_eval_batch_size=train_config['eval_batch_size'],
            dataloader_drop_last=False,
            remove_unused_columns=False,
        )
        
        trainer = CustomTrainer(
            tokenizer=tokenizer,
            input_max_length=train_config['input_max_length'],
            output_max_length=train_config['output_max_length'],
            enable_save_model=False,
            model=model,
            model_type=train_config['model_type'],
            args=training_args
        )
        
        print("开始评估...")
        print("="*20 + "测试集最终指标" + "="*20)
        trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        


if __name__ == "__main__":
    config = yaml.safe_load(open('config.yml', 'r'))
    main(config)

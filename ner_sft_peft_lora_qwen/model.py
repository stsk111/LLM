# !/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from transformers import GenerationConfig, Trainer
from rouge_chinese import Rouge
import jieba
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import collections
import re
import os
import json
from data import TrainCollator, EvalCollator

jieba.initialize()


class CustomTrainer(Trainer):
    def __init__(self, tokenizer, input_max_length, output_max_length, model_type, 
                 enable_save_model=True, *args, **kwargs):
        super().__init__(*args, processing_class=tokenizer, **kwargs)
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.score_dict = {
            'rouge-1':[],
            'rouge-2':[],
            'rouge-l':[],
            'bleu-4':[],
            'F1':[]
        }
        self.best = 0.
        self.model_save_dir = os.path.join(self.args.output_dir, "best_rouge_l")
        self.enable_save_model = enable_save_model
        self.entity_counter = collections.defaultdict(dict)
        self.train_collator = TrainCollator(tokenizer, input_max_length, output_max_length, model_type)
        self.eval_collator = EvalCollator(tokenizer, input_max_length, output_max_length, model_type)

    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            collate_fn=self.train_collator, # <--- 强制指定 TrainCollator
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=True, 
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_collator, # <--- 强制指定 EvalCollator
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            shuffle=False,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        自定义损失计算
        """
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(self, dataloader, description: str, prediction_loss_only=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        自定义评估循环，使用生成模式进行评估
        """
        model = self.model
        model.eval()
        
        # 重置评估指标
        self.score_dict = {
            'rouge-1':[],
            'rouge-2':[],
            'rouge-l':[],
            'bleu-4':[],
            'F1':[]
        }
        
        for batch in dataloader:
            # 手动把数据搬到model.device上，训练过程Trainer内部的_prepare_inputs(inputs)自动搬
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with torch.no_grad():                
                input_ids = batch['input_ids']
                labels = torch.where(batch['labels']==-100, self.processing_class.pad_token_id, batch['labels'])
                attention_mask = batch['attention_mask']
                
                generation_config = GenerationConfig(
                    do_sample=False,
                    eos_token_id=self.processing_class.eos_token_id,
                    pad_token_id=self.processing_class.pad_token_id,
                    return_dict_in_generate=False,
                    output_scores=False,
                    max_new_tokens=self.output_max_length,
                )
                
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
                output_ids = output_ids[...,input_ids.size(1):]

                output_text = self.processing_class.batch_decode(output_ids, skip_special_tokens=True)
                label_text = self.processing_class.batch_decode(labels, skip_special_tokens=True)

                for pred, label in zip(output_text, label_text):
                    print(f"label: {label}")
                    print(f"pred: {pred}\n")

                    if not pred:
                        pred = '-'
                    self.count_entity(pred, label)
                    pred = ' '.join(jieba.cut(pred))
                    label = ' '.join(jieba.cut(label))
                    rouge = Rouge()
                    scores = rouge.get_scores(pred, label)
                    result = scores[0]
                    for k, v in result.items():
                        self.score_dict[k].append(round(v["f"] * 100, 4))
                    bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                    self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))
                    self.score_dict['F1'].append(round(result['rouge-1']['f']*100, 4))
        
        # 计算平均指标
        metrics = collections.defaultdict(float)
        for k,v in self.score_dict.items():
            if v:  
                metrics[f'{metric_key_prefix}_{k}'] = float(np.mean(v))
        
        metrics[f'{metric_key_prefix}_main'] = metrics.get(f'{metric_key_prefix}_rouge-l', 0.0)
        
        # 保存最佳模型
        if self.enable_save_model and metrics[f'{metric_key_prefix}_main'] > self.best:
            self.model.save_pretrained(self.model_save_dir)
            print(f"Model saved to {self.model_save_dir} with score: {metrics[f'{metric_key_prefix}_main']}")
        
        self.best = max(self.best, metrics.get(f'{metric_key_prefix}_main', 0.0))
        
        # 打印实体级别的指标
        print("Entity-level metrics:")
        metrics_entity = {}
        for k,v in self.entity_counter.items():
            precision = float(v['tp']/(v['pred_entity_num'])) if v['pred_entity_num'] > 0 else 0.
            recall = float(v['tp']/(v['label_entity_num'])) if v['label_entity_num'] > 0 else 0.
            f1 = float(2*precision*recall/(precision+recall)) if precision+recall > 0 else 0.
            metrics_entity[k] = {'precision':precision, 'recall':recall, 'f1':f1}
            print(f"{k}: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
        metrics["metrics_entity"] = metrics_entity
        self.entity_counter.clear()

        for k, v in metrics.items():
            print(f"{k}: {json.dumps(v, indent=2)}")

        # 返回transformers期望的格式
        from transformers.trainer_utils import EvalLoopOutput
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
        )

    def count_entity(self, pred, label):
        '''计算每个实体类别的tp,pred_entity_num,label_entity_num'''
        entities_pred = self.parse_entity(pred)
        entities_label = self.parse_entity(label)
        all_names = set(entities_pred.keys()) | set(entities_label.keys())
        for name in all_names:
            counter = self.entity_counter.get(name, {'tp':0, 'pred_entity_num':0 ,'label_entity_num':0})
            counter['tp'] += len([ent for ent in entities_pred.get(name,[]) if ent in entities_label.get(name,[])]) #预测正确的实体数
            counter['pred_entity_num'] += len(entities_pred.get(name,[])) #预测的实体总数
            counter['label_entity_num'] += len(entities_label.get(name,[])) #实际的实体总数
            self.entity_counter[name].update(counter)

    def parse_entity(self, text):
        '''解析实体'''
        #假设文本中的实体都是 XXX:YYY 的形式
        p = r'[^,]+:[^,]+'
        res = collections.defaultdict(list)
        entities = re.findall(p, text)
        for entity in entities:
            ent, name = entity.split(':', 1)
            res[name.strip()].append(ent.strip())
        return res


if __name__ == "__main__":
    run_code = 0

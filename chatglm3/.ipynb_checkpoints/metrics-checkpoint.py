"""
训练过程中使用bleu-4指标来评估模型
"""
from transformers import EvalPrediction
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import jieba
import numpy as np
import torch

def compute_metrics(eval_preds: EvalPrediction):
    batched_pred_ids = torch.from_numpy(eval_preds.predictions[0])
    batched_label_ids = torch.from_numpy(eval_preds.label_ids)

    metrics_dct = {'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_tt = torch.argmax(pred_ids, dim=-1)
        pred_txt = tokenizer.decode(pred_tt).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    return {k: np.mean(v) for k, v in metrics_dct.items()}
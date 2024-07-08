import os

"""
解析命令行参数
"""
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--mode', choices=['test', 'pro'], default='test',
                    help='设置训练模式 (默认: test，只训练小样本，测试脚本功能性)')

# 解析参数
args = parser.parse_args()

"""
公共参数
"""

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf"
os.environ["HF_HUB_CACHE"] = "/root/autodl-tmp/hf"

# 公共参数
storage = '/root/autodl-tmp/'
model_name_or_path = "openai/whisper-large-v2"
model_dir = storage + "models/whisper-large-v2-asr-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"

"""
数据处理 Data Processing
"""
from datasets import load_dataset, DatasetDict
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
from datasets import Audio

common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset(dataset_name, language_abbr, split="validation", trust_remote_code=True)

# 从预训练模型加载特征提取器
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

# 从预训练模型加载分词器，可以指定语言和任务以获得最适合特定需求的分词器配置
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

# 从预训练模型加载处理器，处理器通常结合了特征提取器和分词器，为特定任务提供一站式的数据预处理
processor = AutoProcessor.from_pretrained(model_name_or_path, language=language, task=task)

# 移除不必要的列
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)

# 降采样至16kHz
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


# 抽样数据处理
if args.mode == 'test':
    small_common_voice = DatasetDict()
    small_common_voice["train"] = common_voice["train"].shuffle(seed=16).select(range(640))
    small_common_voice["validation"] = common_voice["validation"].shuffle(seed=16).select(range(320))
    tokenized_common_voice = small_common_voice.map(prepare_dataset)
else:
    # 完整数据训练，尝试开启 `num_proc=8` 参数多进程并行处理（如阻塞无法运行，则不使用此参数）
    tokenized_common_voice = common_voice.map(prepare_dataset, num_proc=8)

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union


# 定义一个针对语音到文本任务的数据整理器类
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # 处理器结合了特征提取器和分词器

    # 整理器函数，将特征列表处理成一个批次
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 从特征列表中提取输入特征，并填充以使它们具有相同的形状
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 从特征列表中提取标签特征（文本令牌），并进行填充
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 使用-100替换标签中的填充区域，-100通常用于在损失计算中忽略填充令牌
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果批次中的所有序列都以句子开始令牌开头，则移除它
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 将处理过的标签添加到批次中
        batch["labels"] = labels

        return batch  # 返回最终的批次，准备好进行训练或评估


# 用给定的处理器实例化数据整理器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

"""
模型加载准备
"""
from transformers import AutoModelForSpeechSeq2Seq

model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
# 设置模型配置中的forced_decoder_ids属性为None
model.config.forced_decoder_ids = None  # 这通常用于指定在解码（生成文本）过程中必须使用的特定token的ID，设置为None表示没有这样的强制要求

# 设置模型配置中的suppress_tokens列表为空
model.config.suppress_tokens = []  # 这用于指定在生成过程中应被抑制（不生成）的token的列表，设置为空列表表示没有要抑制的token

from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

# 创建一个LoraConfig对象，用于设置LoRA（Low-Rank Adaptation）的配置参数
config = LoraConfig(
    r=4,  # LoRA的秩，影响LoRA矩阵的大小
    lora_alpha=64,  # LoRA适应的比例因子
    # 指定将LoRA应用到的模型模块，通常是attention和全连接层的投影。
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,  # 在LoRA模块中使用的dropout率
    bias="none",  # 设置bias的使用方式，这里没有使用bias
)

peft_model = get_peft_model(model, config)
# 打印 LoRA 微调训练的模型参数
peft_model.print_trainable_parameters()

from transformers import Seq2SeqTrainingArguments

"""
训练超参数准备
"""
batch_size = 64
if args.mode == "test":
    num_epoch = 3
else:
    num_epoch = 1

total_steps = len(tokenized_common_voice["train"]) // batch_size * num_epoch
num_logging_entries = 50
num_eval_entries = 10
eval_steps = total_steps // num_eval_entries
logging_steps = total_steps // num_logging_entries
warnup_steps = total_steps // 10
save_limit = 3
save_steps = total_steps // save_limit

# 设置序列到序列模型训练的参数
training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir,  # 指定模型输出和保存的目录
    per_device_train_batch_size=batch_size,  # 每个设备上的训练批量大小
    learning_rate=1e-5,  # 学习率
    num_train_epochs=num_epoch,  # 训练的总轮数
    eval_strategy="steps",  # 设置评估策略
    eval_steps=eval_steps,
    logging_strategy='steps',
    logging_steps=logging_steps,  # 指定日志记录的步骤，用于跟踪训练进度
    warmup_steps=warnup_steps,  # 在训练初期增加学习率的步数，有助于稳定训练
    fp16=True,  # 启用混合精度训练，可以提高训练速度，同时减少内存使用
    per_device_eval_batch_size=batch_size,  # 每个设备上的评估批量大小
    remove_unused_columns=True,  # 是否删除不使用的列，以减少数据处理开销
    label_names=["labels"],  # 指定标签列的名称，用于训练过程中
    # Parameters for early stopping
    save_strategy='steps',
    save_steps=save_steps,
    load_best_model_at_end=True,
    save_total_limit=save_limit,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

"""
训练开始
"""
from transformers import Seq2SeqTrainer

print(f'Training Args: {training_args}')
trainer = Seq2SeqTrainer(
    args=training_args,
    model=peft_model,
    train_dataset=tokenized_common_voice["train"],
    eval_dataset=tokenized_common_voice["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
peft_model.config.use_cache = False

print('training start!')
trainer.train()

for his in trainer.state.log_history:
    print(his)


"""
训练结束，保存模型
"""
print('training finished!')
trainer.save_model(model_dir)
print('model saved!')

log_dir = model_dir + "/runs"
print(f'check out result: tensorboard --logdir {log_dir}')

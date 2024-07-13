import functools
import os

from metrics import compute_metrics

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/hf"
os.environ["HF_HUB_CACHE"] = "/root/autodl-tmp/hf"

"""
全局参数和变量
"""
model_name_or_path = 'THUDM/chatglm3-6b'  # 模型ID或本地路径
train_data_path = 'shibing624/AdvertiseGen'  # 训练数据路径
eval_data_path = None  # 验证数据路径，如果没有则设置为None
seed = 8  # 随机种子
max_input_length = 512  # 输入的最大长度
max_output_length = 1536  # 输出的最大长度
lora_rank = 4  # LoRA秩
lora_alpha = 32  # LoRA alpha值
lora_dropout = 0.05  # LoRA Dropout率
resume_from_checkpoint = None  # 如果从checkpoint恢复训练，指定路径
prompt_text = ''  # 所有数据前的指令文本

"""
数据集预处理
"""
from datasets import load_dataset

dataset = load_dataset(train_data_path)
column_names = dataset['train'].column_names

from transformers import AutoTokenizer
from tokenization import tokenize_func, DataCollatorForChatGLM

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          trust_remote_code=True,
                                          revision='b098244'
                                          )
tokenized_dataset = dataset.map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False,
    remove_columns=column_names,
    num_proc=4,
)

# split tokenized_dataset into train, validation and test sets
train_test_set = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=seed)
train_set = train_test_set['train']
test_set = train_test_set['test']
eval_set = tokenized_dataset['validation']

# 准备数据整理器
data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)

from transformers import BitsAndBytesConfig, AutoModelForCausalLM
import torch

_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

"""
QLoRA加载模型
"""
# QLoRA 量化配置
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])

# revision='b098244' 版本对应的 ChatGLM3-6B 设置 use_reentrant=False
# 最新版本 use_reentrant 被设置为 True，会增加不必要的显存开销
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             quantization_config=q_config,
                                             device_map='auto',
                                             trust_remote_code=True,
                                             revision='b098244'
                                             )

from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_kbit_training

kbit_model = prepare_model_for_kbit_training(model)

from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

lora_config = LoraConfig(
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)

qlora_model = get_peft_model(kbit_model, lora_config)

"""
训练超参数配置
"""
from transformers import TrainingArguments, Trainer

save_dir = '/root/autodl-tmp/models'
step_ratio = 0.1

training_args = TrainingArguments(
    overwrite_output_dir=True,
    output_dir=f"{save_dir}/{model_name_or_path}",  # 输出目录
    per_device_train_batch_size=16,  # 每个设备的训练批量大小
    per_device_eval_batch_size=16,  # 每个设备的评估批量大小
    gradient_accumulation_steps=4,
    eval_accumulation_steps=2,
    learning_rate=1e-3,  # 学习率
    lr_scheduler_type="linear",  # 学习率调度器类型
    warmup_ratio=step_ratio,  # 预热比例
    logging_steps=step_ratio,  # 日志记录步数
    evaluation_strategy="steps",  # 评估策略
    eval_steps=step_ratio,  # 评估步数
    fp16=True,  # 是否使用混合精度训练
    num_train_epochs=1,  # 训练轮数
    report_to=['tensorboard'],
    save_strategy='steps',
    save_steps=0.1,
)

trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set.select(range(300)),
    data_collator=data_collator,
    compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
)

trainer.train()
trainer.save_model()
trainer.save_state()

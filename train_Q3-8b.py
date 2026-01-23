import os
# 强制离线模式：不访问任何Hugging Face网络
os.environ["HF_HUB_OFFLINE"] = "true"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 禁用Unsloth的快速下载功能（避免它尝试访问外网）
os.environ["UNSLOTH_FAST_DOWNLOAD"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset,Dataset,concatenate_datasets
from swanlab.integration.transformers import SwanLabCallback
import pandas as pd
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from peft import LoraConfig, TaskType, get_peft_model

from typing import Optional, List, Union
import sys
import deepspeed


# DS_CONFIG = "/mnt/nvme1n1/wjc/Github/Mental/SoulChat-main/train_model_paralell/ds_zero2_no_offload.json"
DS_CONFIG = "/mnt/nvme1n1/wjc/Github/Mental/SoulChat-main/train_model_paralell/ds_z2_offload_config.json"

model_name = "/mnt/nvme1n1/wjc/Model/Qwen3-8B-merged/Qwen3-8B-merged-EmoLLM"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,  # 比如Qwen/Qwen-7B-Chat等
    padding=True,   # 自回归模型优先右padding
    truncation=True,
    use_fast=True,            # 加速tokenize
    trust_remote_code=True,    # 自定义模型需加
)

device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    # device_map=device_map,
    # device_map="auto"
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=16,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.05,  # Dropout 比例
)

model = get_peft_model(model, config)

# ds_reason = load_dataset("Ronndy/medical_o1_sft_Chinese",cache_dir='/mnt/nvme1n1/wjc/Datasets/Ronndy-medical_o1_sft_Chinese',split="train[:1600]")
ds1 = load_dataset(
    "json",  # 数据集格式（如果保存的是JSON，这里改为"json"）
    data_files="/mnt/nvme1n1/wjc/My_dataset/train/train_data_con_audio.json",
    cache_dir='/mnt/nvme1n1/wjc/My_dataset/train/',  # 缓存目录（可选，用于保存数据集元数据）
    split="train"
)

ds2 = load_dataset(
    "json",
    data_files="/mnt/nvme1n1/wjc/My_dataset/train/train_data3_.json",
    cache_dir='/mnt/nvme1n1/wjc/My_dataset/train/',
    split="train"
)


ds3 = load_dataset(
    "json",  # 数据集格式（如果保存的是JSON，这里改为"json"）
    data_files="/mnt/nvme1n1/wjc/My_dataset/train/train_data_con_en.json",
    cache_dir='/mnt/nvme1n1/wjc/My_dataset/train/',  # 缓存目录（可选，用于保存数据集元数据）
    split="train"
)

ds4 = load_dataset(
    "json",
    data_files="/mnt/nvme1n1/wjc/My_dataset/train/train_data_con_en2.json",
    cache_dir='/mnt/nvme1n1/wjc/My_dataset/train/',
    split="train"
)


merged_ds = concatenate_datasets([ds1, ds2, ds3, ds4])
merged_ds = merged_ds.shuffle(seed=3407)  # 先打乱

def convert_to_chat_format(examples):
    conversations = []
    for msg_list in examples["messages"]:
        conv = []
        # 提取system提示词
        system_prompt = next((msg["content"] for msg in msg_list if msg["role"] == "system"), "")
        if system_prompt:
            conv.append({"role": "system", "content": system_prompt})
        # 按顺序收集user和assistant的多轮对话
        for msg in msg_list:
            if msg["role"] in ["user", "assistant"]:
                conv.append({"role": msg["role"], "content": msg["content"]})
        conversations.append(conv)
    return {"conversations": conversations}

# 批量处理数据集
ds_alpaca = merged_ds.map(
    convert_to_chat_format,
    batched=True,
    remove_columns=merged_ds.column_names,  # 移除原有的"conversation"列
)


ds_alpaca_valid = ds_alpaca.filter(lambda x: len(x["conversations"]) > 0)

def process_single_conv(example):
    processed_text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        role_key="role",
        content_key="content",
    )
    return {"text": processed_text}  # 直接生成"text"列，用于SFTTrainer

combined_dataset = ds_alpaca_valid.map(
    process_single_conv,
    remove_columns=["conversations"]  # 移除不需要的列
)

# 数据清洗：过滤掉长度小于5的无效数据（直接在Dataset上过滤）
print(f"原始数据量: {len(combined_dataset)}")
combined_dataset = combined_dataset.filter(
    lambda x: len(x["text"]) > 5  # 过滤条件：文本长度大于5
)
print(f"清洗后数据量: {len(combined_dataset)}")

# 随机打乱数据集（保持原随机种子）
combined_dataset = combined_dataset.shuffle(seed=3407)

# 验证：打印一条数据看看格式对不对
print("示例数据片段:", combined_dataset[0]["text"][:100])

# # 将转换后的推理数据集应用对话模板
# combined_dataset = tokenizer.apply_chat_template(
#     ds_alpaca_valid["conversations"],  # 用过滤后的数据集
#     tokenize=False,
#     role_key="role",  # 对应样本里的"role"字段
#     content_key="content",  # 对应样本里的"content"字段
#     chat_template=None,  # 使用Qwen3内置的聊天模板（自动匹配user/assistant）
#     add_generation_prompt=False  # 不需要生成式前缀（若要生成可设为True）
# )


# dataset_final = Dataset.from_dict({
#     "text": combined_dataset  # 用text字段存储渲染后的对话模板
# })

# dataset_final = combined_dataset.select_columns(["text"]).shuffle(seed=3407)
# 此时又可以用Dataset的shuffle方法（可选）
combined_dataset



swanlab_callback = SwanLabCallback(
    project="Qwen3-8B-final",
    experiment_name="Qwen3-8B-emo+four_data-r16-b2",
    description="使用通义千问Qwen3-8B-merged-Emo+four_data模型在train_data_con_audio.json、train_data3_.json、train_data_con_en.json和train_data_con_en2.json数据集上微调。",
    config={
        "model": "/mnt/nvme1n1/wjc/Model/Qwen3-8B-merged/Qwen3-8B-merged-EmoLLM",
        "dataset": "/mnt/nvme1n1/wjc/My_dataset/train_data_con_audio.json",
        "train_data_number": len(combined_dataset),
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
)

# export CUDA_VISIBLE_DEVICES=5,7

sft_config = SFTConfig(
    output_dir="/mnt/nvme1n1/wjc/Github/Mental/SoulChat-main/lora_model/Qwen3-8B-emo+four_data",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=20,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=5,
    optim="adamw_torch", 
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    bf16=True,
    fp16=False,
    max_grad_norm=1.0,
    deepspeed=DS_CONFIG,
    logging_first_step=True,  # 修正：原代码写的5是错误的，应为布尔值
    save_steps=100,
    dataset_text_field="text",  # 关键：dataset_text_field移到这里
    max_length=4096,        # 新增：限制序列长度（根据显存调整）
    packing=False,              # 关闭packing（多轮对话不建议开启）
    remove_unused_columns=False,# 保留数据集列
    load_best_model_at_end=False# 无验证集时关闭
)

# 7. 初始化SFTTrainer（修正参数名）
trainer = SFTTrainer(
    model=model,
    # dataset_text_field="text",  # 关键：dataset_text_field移到这里
    processing_class=tokenizer,  # 修正：替换旧的processing_class
    train_dataset=combined_dataset,
    eval_dataset=None,
    args=sft_config,      # 传入配置对象
    # max_seq_length=4096,        # 新增：限制序列长度（根据显存调整）
    callbacks=[swanlab_callback]
)


# 显示当前内存统计
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# 显示最终内存和时间统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")

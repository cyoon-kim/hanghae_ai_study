import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers
import re

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

wandb.init(project='week7_basic')
wandb.run.name = 'Gemma-basic-finetuning'

torch.cuda.empty_cache()

# 모델과 데이터셋을 선택한 이유
# 1. 모델 : google/gemma-3-1b-pt (https://huggingface.co/google/gemma-3-1b-pt)
#    - Google에서 개발한 LLM으로, 1B 파라미터를 가진 모델입니다.
#    - task에 따라 it보다는 pt계열 모델 사용용
# 2. 데이터셋 : wikitext-2-raw-v1 (https://huggingface.co/datasets/wikitext)
#    - 영어 위키피디아의 raw text로 구성되어 있습니다.
#    - 다양한 주제를 포함하고 있어, 일반적인 언어 모델링에 적합합니다.
# 3. task : next token prediction

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="google/gemma-3-1b-pt")  # HuggingFace hub에서 pre-trained 모델로 사용할 모델의 이름
    torch_dtype: Optional[str] = field(default='bfloat16', metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})  # 우리 모델의 precision(data type이라고 이해하시면 됩니다)

    dataset_name: Optional[str] = field(default="wikitext")  # Fine-tuning으로 사용할 huggingface hub에서의 dataset 이름
    dataset_config_name: Optional[str] = field(default="wikitext-2-raw-v1")  # Fine-tuning으로 사용할 huggingface hub에서의 dataset configuration
    block_size: int = field(default=256)  # Fine-tuning에 사용할 input text의 길이
    num_workers: Optional[int] = field(default=2)  # Data를 업로드하거나 전처리할 때 사용할 worker 숫자
    
parser = HfArgumentParser(Arguments)
args = parser.parse_args_into_dataclasses()[0]

logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# 데이터 확인 시 wiki 데이터셋이라 불필요한 특수문자나 기호가 포함되어 있어 이를 제거하기 위해 사용
def clean_text(example):
    text = example["text"].strip()
    text = text.replace(" @-@ ", "-")
    example["text"] = text
    return example

def filter_valid(example):
    text = example["text"].strip()
    return text != "" and not re.match(r"^=+ .+ =+$", text)

raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

for split in raw_datasets:
    raw_datasets[split] = raw_datasets[split].map(clean_text)
    raw_datasets[split] = raw_datasets[split].filter(filter_valid)

# 모델 설정
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype,
    attn_implementation='eager'
)

# gpt 유형 모델 pad token을 eos token으로 설정
tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    output = tokenizer(examples['text'])
    return output

def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // args.block_size) * args.block_size
    result = {
        k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=args.num_workers,
    remove_columns=raw_datasets["train"].column_names
)

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=args.num_workers,
)

lm_datasets["train"] = lm_datasets["train"].select(range(100))
lm_datasets["validation"] = lm_datasets["validation"].select(range(20))
lm_datasets["test"] = lm_datasets["test"].select(range(20))

print("** Dataset size after tokenization and grouping:")
print(lm_datasets['train'])
print(f"train dataset size: {len(lm_datasets['train'])}")
print(f"validation dataset size: {len(lm_datasets['validation'])}")
print(f"test dataset size: {len(lm_datasets['test'])}")

training_args = TrainingArguments(
    output_dir="./gemma-finetuned",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    eval_strategy="epoch",
    save_strategy="no",
    per_device_eval_batch_size=1,
    logging_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs",
    report_to="wandb"   ,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=False,
    learning_rate=1e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

before_results = trainer.evaluate(lm_datasets["test"], metric_key_prefix="init_test")
print(f"Initial test results: {before_results}")

    
train_result = trainer.train()
print(f"Training completed. Final results: {train_result}")

trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

after_results = trainer.evaluate(lm_datasets["test"], metric_key_prefix="final_test")
print(f"Final test results: {after_results}")
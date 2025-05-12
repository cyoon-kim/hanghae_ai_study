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

# wandb.init(project='week7_basic')
# wandb.run.name = 'gpt-finetuning'

# 모델과 데이터셋을 선택한 이유
# 1. 모델 : gemma-3-1b-pt (https://huggingface.co/google/gemma-3-1b-pt)
#    - 1.5B 파라미터를 가진 모델로, 적당한 크기와 성능을 가지고 있습니다.
#    - pre-trained 모델로, fine-tuning을 통해 성능을 향상시킬 수 있습니다.
#    - task에 맞게 it보다는 pt로 fine-tuning을 진행합니다.
# 2. 데이터셋 : wikitext-2-raw-v1 (https://huggingface.co/datasets/wikitext)
#    - 영어 위키피디아의 raw text로 구성되어 있습니다.
#    - 다양한 주제를 포함하고 있어, 일반적인 언어 모델링에 적합합니다.
# 3. task : next token prediction

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default="google/gemma-3-1b-pt")  # HuggingFace hub에서 pre-trained 모델로 사용할 모델의 이름
    torch_dtype: Optional[str] = field(default='float16', metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})  # 우리 모델의 precision(data type이라고 이해하시면 됩니다)

    dataset_name: Optional[str] = field(default="wikitext")  # Fine-tuning으로 사용할 huggingface hub에서의 dataset 이름
    dataset_config_name: Optional[str] = field(default="wikitext-2-raw-v1")  # Fine-tuning으로 사용할 huggingface hub에서의 dataset configuration
    block_size: int = field(default=1024)  # Fine-tuning에 사용할 input text의 길이
    num_workers: Optional[int] = field(default=2)  # Data를 업로드하거나 전처리할 때 사용할 worker 숫자
    
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# logger = logging.getLogger()

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )

# if training_args.should_log:
#     transformers.utils.logging.set_verbosity_info()  # log level을 INFO로 변경 

# log_level = training_args.get_process_log_level()

# # 우리가 가지고 있는 logger와 HuggingFace의 logger의 log level 설정
# logger.setLevel(log_level)
# datasets.utils.logging.set_verbosity(log_level)
# transformers.utils.logging.set_verbosity(log_level)

# # 기타 HuggingFace logger option들을 설정
# transformers.utils.logging.enable_default_handler()
# transformers.utils.logging.enable_explicit_format()

# logger.info(f"Training/evaluation parameters {training_args}")

# 데이터 확인 시 wiki 데이터셋이라 불필요한 특수문자나 기호가 포함되어 있어 이를 제거하기 위해 사용
def clean_text(example):
    text = example["text"].strip()
    text = text.replace(" @-@ ", "-")
    example["text"] = text
    return example

def filter_valid(example):
    text = example["text"].strip()
    return text != "" and not re.match(r"^=+ .+ =+$", text)

data_ratio = 0.02  # 사용할 데이터 비율

raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

data_ratio = 0.02

for split in raw_datasets:
    raw_datasets[split] = raw_datasets[split].map(clean_text)
    raw_datasets[split] = raw_datasets[split].filter(filter_valid)

    total = len(raw_datasets[split])
    n = int(total * data_ratio)
    raw_datasets[split] = raw_datasets[split].select(range(n))
    print(f"{split}: {n}개 샘플 사용")

# 모델 설정
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

tokenizer.pad_token_id = tokenizer.eos_token_id

# embedding_size = model.get_input_embeddings().weight.shape[0]
# if len(tokenizer) > embedding_size:
#     model.resize_token_embeddings(len(tokenizer))

# column_names = list(raw_datasets["train"].features)
# text_column_name = "text" if "text" in column_names else column_names[0]

# def tokenize_function(examples):
#     output = tokenizer(examples[text_column_name])
#     return output
    
# with training_args.main_process_first(desc="dataset map tokenization"):
#     tokenized_datasets = raw_datasets.map(
#         tokenize_function,
#         batched=True,
#         num_proc=args.num_workers,
#         remove_columns=column_names
#     )

# max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
# block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

# def group_texts(examples):
#     # 주어진 text들을 모두 concat 해줍니다. 
#     # 예를 들어 examples = {'train': [['Hello!'], ['Yes, that is great!']]}이면 결과물은 {'train': ['Hello! Yes, that is great!']}가 됩니다.
#     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    
#     # 전체 길이를 측정합니다.
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     total_length = (total_length // block_size) * block_size
    
#     # block_size로 text를 쪼갭니다.
#     # 예를 들어 block_size=3일 때 {'train': ['Hello! Yes, that is great!']}는
#     # {'train': ['Hel', 'lo!', ' Ye', 's, ', 'tha', ...]}가 됩니다. 
#     result = {
#         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#         for k, t in concatenated_examples.items()
#     }
    
#     # Next token prediction이니 label은 자기 자신으로 설정합니다.
#     result["labels"] = result["input_ids"].copy()
#     return result
    
# with training_args.main_process_first(desc="grouping texts together"):
#     lm_datasets = tokenized_datasets.map(
#         group_texts,
#         batched=True,
#         num_proc=args.num_workers
#     )
    
# train_dataset = lm_datasets["train"]

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer,
#     data_collator=default_data_collator
# )

# checkpoint = None
# last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
# if training_args.resume_from_checkpoint is not None:  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
#     checkpoint = training_args.resume_from_checkpoint
# else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.  
#     checkpoint = last_checkpoint
    
# train_result = trainer.train(resume_from_checkpoint=checkpoint)

# trainer.save_model()

# metrics = train_result.metrics
# trainer.log_metrics("train", metrics)
# trainer.save_metrics("train", metrics)
# trainer.save_state()
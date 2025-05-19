import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import torch
from tqdm import tqdm
import os

model_name = "google/gemma-3-1b-it"

wandb.init(project='week7_advanced')
wandb.run.name = 'Gemma-advanced-finetuning'

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# corpus.json 불러오기
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

train_data, temp_data = train_test_split(corpus, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)  # 1/3 of 30% = 10%

dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test": Dataset.from_list(test_data),
})

print("** Dataset size after tokenization and grouping:")
print(dataset)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def format_and_tokenize(example):
    prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    
    tokenized = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(format_and_tokenize, batched=False)

print("finish preparing data")
print("--------------")
# ✅ 모델 로딩 + LoRA 적용
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    device_map="auto"
)
base_model.config.use_cache = False
base_model.gradient_checkpointing_enable()

# ✅ LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Gemma 구조 기준
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # ✅ 확인용 (LoRA만 학습 대상인지)

# ✅ TrainingArguments
training_args = TrainingArguments(
    output_dir="./gemma-lora-finetuned",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="no",
    save_total_limit=1,
    report_to="wandb",
    run_name="gemma-lora-tuning",
    learning_rate=1e-4,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# 평가 함수
def make_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:"""

def eval_model(model):
    model.eval()
    results = []
    for example in tqdm(dataset["test"]):
        prompt = make_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=True, 
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_output = decoded_output.split("### Response:")[-1].strip()

        results.append({
            "input": example["input"],
            "expected_output": example["output"],
            "model_output": model_output
        })
    return results

# init_result = eval_model(model)

# with open("init_outputs.json", "w", encoding="utf-8") as f:
#     json.dump(init_result, f, ensure_ascii=False, indent=2)

trainer.train()
trainer.save_model()

after_train_result = eval_model(trainer.model)

with open("after_outputs.json", "w", encoding="utf-8") as f:
    json.dump(after_train_result, f, ensure_ascii=False, indent=2)
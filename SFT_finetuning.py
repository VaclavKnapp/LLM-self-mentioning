import torch
import os
from datasets import load_dataset
import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments


import wandb

os.environ["WANDB_PROJECT"] = "qwen2.5-entity-extraction" 
os.environ["WANDB_LOG_MODEL"] = "false" 


model_name = "unsloth/Qwen2.5-7B"
dataset_path = "/home/vaclav_knapp/Memory_Bank_VLM/LLM_part/datasets/self-reference_sft_data.jsonl"
output_dir = "fine_tuned_qwen_sft_2"
max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0, 
    bias="none", 
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
    use_rslora=False,  
    loftq_config=None, 
)

def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    completions = examples["completion"]
    texts = []
    EOS_TOKEN = tokenizer.eos_token
    
    system_instruction = (
        "You are a script analysis expert. Classify name mentions in the dialogue. "
        "Format the output as {Category: Name}. "
        "Use 'Self-reference' if the speaker is mentioning their own name. "
        "Use 'else' if the speaker is mentioning someone else."
        "If there is no name present format the output as {none}"
    )

    for prompt, completion in zip(prompts, completions):

        clean_prompt = prompt.replace("Classify the following script line for name mentions: ", "")
        
        text = (
            f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            f"<|im_start|>user\n{clean_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{completion}<|im_end|>"
        ) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files=dataset_path, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False, 
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=200,
        num_train_epochs=15, 
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        

        report_to="wandb",          
        run_name="qwen-entity-run-1", 
        save_strategy="steps",      
        save_steps=50,              
    ),
)


trainer_stats = trainer.train()

wandb.finish()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuning complete. Model saved to {output_dir}.")

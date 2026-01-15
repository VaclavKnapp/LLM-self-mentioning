import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from tqdm import tqdm
import random

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def generate_response(messages, max_tokens=100):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9
    )
    generated_ids = generated_ids[0][len(inputs['input_ids'][0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

csv_path = "/home/vaclav_knapp/Memory_Bank_VLM/LLM_part/datasets/TBBT.csv"
df = pd.read_csv(csv_path)

sft_data = []
none_data = []  # Collect all cases where no names are extracted

for index, row in tqdm(df.iterrows(), total=len(df)):
    speaker = str(row.get('person_scene', ''))
    text = str(row.get('dialogue', ''))
    if not text or speaker.lower() == 'scene' or not speaker:
        continue
    #print(f"Text: {text}")
    #print(f"Speaker: {speaker}")
    extract_messages = [
        {"role": "system", "content": """You are an expert named entity recognition system specialized in extracting only proper names of specific persons from text. A proper name is a capitalized noun or noun phrase that uniquely identifies an individual person, such as 'Chandler', 'Rachel Green', 'Dr. Ross Geller', or 'Joe'. It must refer to a distinct human being, not a group, object, place, or abstract concept.
Rules and constraints:
- Extract ONLY proper names of persons. Do not extract pronouns (e.g., 'I', 'he', 'she', 'you', 'me', 'him', 'her', 'we', 'they', 'it').
- Do not extract relational or generic terms (e.g., 'my mother', 'his father', 'the guy', 'some friend', 'man', 'woman', 'someone', 'everybody', 'dad', 'mom', 'brother', 'sister', 'Daddy').
- Do not extract titles alone (e.g., 'Mr.', 'Dr.', 'Professor') unless they are part of a full proper name (e.g., 'Mr. Smith').
- Ignore possessives unless the core name is proper (e.g., extract 'John' from "John's book", but not 'mother' from "my mother's call").
- Names are typically capitalized in English text; ignore uncapitalized words unless clearly a name in context.
- Extract unique names only; do not duplicate.
- If no proper person names are present, respond exactly with 'none'.
- Don not extract names of places or Not-living things (e.g. Haiti, White House, Port-au-Prince)        
Think step-by-step before responding:
1. Read the entire text carefully.
2. Identify all capitalized words or phrases that could be names.
3. For each candidate, check if it uniquely refers to a specific person (not a relation or generic).
4. Exclude any that match the prohibited categories (pronouns, relations, generics).
5. Compile the final list of unique proper names.
Respond ONLY with the comma-separated list of extracted names (e.g., 'Chandler, Rachel') or exactly 'none'. No explanations, no additional text, no sentences.
Examples:
- Text: "All of a sudden, the phone starts to ring. And it turns out it's my mother, which is very weird, because- she never calls me!"
  Response: none
- Text: "Ever since she walked out on me, I, uh..."
  Response: none
- Text: "Hi, I'm Chandler Bing."
  Response: Chandler Bing
- Text: "Rachel said to Monica, 'Hello friend.'"
  Response: Rachel, Monica
- Text: "My friend Joe and his dad went to see Dr. Smith."
  Response: Joe, Dr. Smith
- Text: "The man called his brother."
  Response: none
- Text: "Please answer, it's me"
  Response: none
- Text: "Can I help you?"
  Response: none
- Text: "President Lincoln gave a speech."
  Response: President Lincoln""" },
        {"role": "user", "content": f"Extract proper names of people from this text: {text}"}
    ]
    
    extracted = generate_response(extract_messages, max_tokens=50)
    #print(f"Extracted_name: {extracted}")
    
    
    if extracted.lower() == 'none':
        sft_prompt = f"Classify the following script line for name mentions: {text}"
        none_data.append({"prompt": sft_prompt, "completion": "{none}"})
        continue
    
    names = [name.strip() for name in extracted.split(',') if name.strip()]
    if len(names) > 1:
        continue
    
    self_names = []
    other_names = []
    for name in names:
        classify_messages = [
            {"role": "system", "content": """You are an expert dialogue reference classifier. Given the speaker's proper name, a mentioned proper name, and the text, classify if the mentioned name refers to the speaker (self) or another person (other).
Rules:
- Classify as 'self' only if the text clearly indicates the speaker is referring to themselves with that name and the names are same (e.g. 'Mickey' for 'Michael', 'John' for 'Johny').
- If the mentioned name matches or is a variant of the speaker's name (e.g., 'Chandler' for 'Chandler Bing'), classify 'self'.
- Otherwise, classify 'other', even if names match but context is about someone else.
- Think step-by-step: 1. Compare names. 2. Decide.
Respond ONLY with 'self' or 'other'. No extra text.
Examples:
- Speaker: 'Chandler', Name: 'Chandler', Text: "Hi, I'm Chandler."
  Response: self
- Speaker: 'Chandler', Name: 'Monica', Text: "Monica is my friend."
  Response: other
- Speaker: 'Paul', Name: 'Rachel', Text: "Rachel left me."
  Response: other
- Speaker: 'Rachel', Name: 'Rachel', Text: "Rachel here, speaking."
  Response: self
- Speaker: 'Joe', Name: 'Joe', Text: "Did you see what Joe did?"
  Response: other""" },
            {"role": "user", "content": f"Speaker's name: '{speaker}'. Mentioned name: '{name}'. Text: '{text}'."}
        ]
        classification = generate_response(classify_messages, max_tokens=20).lower()
        #print(f"Classification: {classification}")
        if classification == 'self':
            self_names.append(name)
        elif classification == 'other':
            other_names.append(name)
    
    if not self_names and not other_names:
        continue
    
    output_parts = []
    if self_names:
        output_parts.append(f"Self-reference: {', '.join(self_names)}")
    if other_names:
        output_parts.append(f"else: {', '.join(other_names)}")
    output_format = "{" + "; ".join(output_parts) + "}"
    
    sft_prompt = f"Classify the following script line for name mentions: {text}"
    sft_data.append({"prompt": sft_prompt, "completion": output_format})

# Determine the number of none entries to add: 20% of the number of positive entries in sft_data
num_positive = len(sft_data)
num_to_select = int(0.2 * num_positive)
random.seed(42)  # For reproducibility
selected_none = random.sample(none_data, min(num_to_select, len(none_data))) if none_data else []
sft_data.extend(selected_none)

jsonl_path = "self-reference_TBBT_eval_data.jsonl"
with open(jsonl_path, 'w') as f:
    for entry in sft_data:
        f.write(json.dumps(entry) + '\n')

print(f"Preprocessing complete. {len(sft_data)} entries written to {jsonl_path}.")

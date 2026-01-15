import torch
from unsloth import FastLanguageModel
import json
from tqdm import tqdm
import re
from datetime import datetime
import random

model_path = "fine_tuned_qwen_sft_2"
max_seq_length = 2048
dtype = None  
load_in_4bit = False


USE_BALANCED_SUBSET = True

# Percentage of "none" examples relative to self-reference count
# Set to -1 to have the same number of "none" as self-reference
# Set to 0.5 for 50% of self-reference count, 1.0 for 100%, 2.0 for 200%, etc.
NONE_PERCENTAGE = -1

# Random seed for reproducibility when sampling
RANDOM_SEED = 42
# =============================================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)

system_instruction = (
    "You are a script analysis expert. Classify name mentions in the dialogue. "
    "Format the output as {Category: Name}"
    "Use 'Self-reference' if the speaker is mentioning their own name. "
    "Use 'else' if the speaker is mentioning someone else."
    "If there is no name present format the output as {none}"
)

def format_input_prompt(script_line):
    clean_prompt = script_line.replace("Classify the following script line for name mentions: ", "")
    formatted_prompt = (
        f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        f"<|im_start|>user\nClassify the following script line for name mentions: {clean_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return formatted_prompt

def generate_classification(script_line):
    formatted_prompt = format_input_prompt(script_line)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,  
        do_sample=False,    
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_text = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return generated_text.strip()

def parse_output(output):
    output = re.sub(r'\s+', ' ', output.strip())
    match = re.search(r'\{(.*?)\}', output)
    if not match:
        return None, set()
    content = match.group(1).strip()
    if content.lower() == 'none':
        return 'none', set()
    parts = [p.strip() for p in content.split(';')]
    mentions = set()
    for part in parts:
        if ':' in part:
            category, name_str = part.split(':', 1)
            category = category.strip().lower()
            name_str = re.sub(r'[}\.,;]+$', '', name_str.strip())
            names = [re.sub(r'[^\w\s-]', '', n.strip()).lower() for n in name_str.split(',')]
            for name in names:
                if name:
                    mentions.add((category, name))
    return 'mentions', mentions

def get_primary_category(entry):
    """Determine the primary category of an entry based on its ground truth."""
    ground_truth = entry['completion']
    gt_type, gt_mentions = parse_output(ground_truth)
    
    if gt_type == 'none':
        return 'none'
    elif gt_mentions:
        return list(gt_mentions)[0][0]
    return 'unknown'

def create_balanced_subset(data, none_percentage=-1, random_seed=42):
    """
    Create a balanced subset of the data.
    
    Args:
        data: List of all data entries
        none_percentage: Percentage of "none" examples relative to self-reference count.
                        Set to -1 to have the same number as self-reference.
        random_seed: Random seed for reproducibility
    
    Returns:
        Balanced subset of data
    """
    random.seed(random_seed)

    self_reference_data = []
    else_data = []
    none_data = []
    
    for entry in data:
        category = get_primary_category(entry)
        if category == 'self-reference':
            self_reference_data.append(entry)
        elif category == 'else':
            else_data.append(entry)
        elif category == 'none':
            none_data.append(entry)
    
    # Get counts
    num_self_reference = len(self_reference_data)
    
    print(f"\nOriginal dataset distribution:")
    print(f"  Self-reference: {num_self_reference}")
    print(f"  Else: {len(else_data)}")
    print(f"  None: {len(none_data)}")
    

    subset = self_reference_data.copy()
    
    # Sample else examples (same count as self-reference)
    num_else_to_sample = min(num_self_reference, len(else_data))
    if len(else_data) > num_self_reference:
        sampled_else = random.sample(else_data, num_else_to_sample)
    else:
        sampled_else = else_data.copy()
    subset.extend(sampled_else)
    
    # Sample none examples based on percentage
    if none_percentage == -1:
        num_none_to_sample = num_self_reference
    else:
        num_none_to_sample = int(num_self_reference * none_percentage)
    
    num_none_to_sample = min(num_none_to_sample, len(none_data))
    if len(none_data) > num_none_to_sample:
        sampled_none = random.sample(none_data, num_none_to_sample)
    else:
        sampled_none = none_data.copy()
    subset.extend(sampled_none)
    
    # Shuffle the subset
    random.shuffle(subset)
    
    print(f"\nBalanced subset distribution:")
    print(f"  Self-reference: {num_self_reference}")
    print(f"  Else: {len(sampled_else)}")
    print(f"  None: {len(sampled_none)}")
    print(f"  Total: {len(subset)}\n")
    
    return subset, {
        'original': {
            'self-reference': len(self_reference_data),
            'else': len(else_data),
            'none': len(none_data),
            'total': len(data)
        },
        'subset': {
            'self-reference': num_self_reference,
            'else': len(sampled_else),
            'none': len(sampled_none),
            'total': len(subset)
        }
    }

if __name__ == "__main__":
    test_jsonl_path = "/home/vaclav_knapp/names_binding/Memory_Bank_VLM/LLM_part/datasets/self-reference_TBBT_eval_data.jsonl" 
    data = []
    with open(test_jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    # Create balanced subset if enabled
    if USE_BALANCED_SUBSET:
        data, distribution_info = create_balanced_subset(data, NONE_PERCENTAGE, RANDOM_SEED)
    else:
        distribution_info = None
    
    total = len(data)
    correct = 0
    category_stats = {'none': {'total': 0, 'correct': 0},
                      'self-reference': {'total': 0, 'correct': 0},
                      'else': {'total': 0, 'correct': 0}}
    
    # Lists to store examples for analysis
    true_positives = []   # Correctly identified mentions (self-reference or else)
    false_positives = []  # Predicted a mention when ground truth was none, or wrong category/name
    false_negatives = []  # Predicted none when there was a mention, or missed mentions
    
    # Store all results for detailed logging
    all_results = []
    
    for entry in tqdm(data):
        prompt = entry['prompt']
        ground_truth = entry['completion']
        
        generated = generate_classification(prompt)
        

        gen_type, gen_mentions = parse_output(generated)
        gt_type, gt_mentions = parse_output(ground_truth)
    
        if gt_type == 'none':
            primary_category = 'none'
        else:

            if gt_mentions:
                primary_category = list(gt_mentions)[0][0]
            else:
                primary_category = 'unknown'  
        
        category_stats[primary_category]['total'] += 1
        
        # Create result entry
        result_entry = {
            'prompt': prompt,
            'ground_truth': ground_truth,
            'generated': generated,
            'gt_type': gt_type,
            'gen_type': gen_type,
            'gt_mentions': gt_mentions,
            'gen_mentions': gen_mentions,
            'primary_category': primary_category
        }
        
        # Compare: for 'none', exact type match
        # For mentions, check if sets match (ignores order, case, extra spaces)
        is_correct = False
        if gen_type == gt_type:
            if gt_type == 'none' or gen_mentions == gt_mentions:
                correct += 1
                category_stats[primary_category]['correct'] += 1
                is_correct = True
        
        result_entry['is_correct'] = is_correct
        all_results.append(result_entry)
        
        # Categorize into TP, FP, FN
        # True Positive: Ground truth has mentions AND prediction correctly identifies them
        # False Positive: Prediction has mentions that are wrong or when GT is none
        # False Negative: Ground truth has mentions but prediction misses them or says none
        
        if gt_type == 'mentions' and gen_type == 'mentions' and gen_mentions == gt_mentions:
            # True Positive - correctly identified all mentions
            if len(true_positives) < 20:
                true_positives.append(result_entry)
        elif gt_type == 'none' and gen_type == 'mentions':
            # False Positive - predicted mentions when there were none
            if len(false_positives) < 20:
                false_positives.append(result_entry)
        elif gt_type == 'mentions' and gen_type == 'mentions' and gen_mentions != gt_mentions:
            # Partial match - could be FP (extra predictions) or FN (missed predictions)
            if len(false_positives) < 20:
                false_positives.append(result_entry)
        elif gt_type == 'mentions' and gen_type == 'none':
            # False Negative - missed mentions entirely
            if len(false_negatives) < 20:
                false_negatives.append(result_entry)
        elif gt_type == 'mentions' and gen_type is None:
            # False Negative - failed to parse/generate proper output
            if len(false_negatives) < 20:
                false_negatives.append(result_entry)
    
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation complete on {total} examples.")
    

    self_acc = category_stats['self-reference']['correct'] / category_stats['self-reference']['total'] * 100 if category_stats['self-reference']['total'] > 0 else 0
    print(f"Self-reference Accuracy: {self_acc:.2f}% ({category_stats['self-reference']['correct']}/{category_stats['self-reference']['total']})")
    

    else_acc = category_stats['else']['correct'] / category_stats['else']['total'] * 100 if category_stats['else']['total'] > 0 else 0
    print(f"Else-reference Accuracy: {else_acc:.2f}% ({category_stats['else']['correct']}/{category_stats['else']['total']})")
    

    none_acc = category_stats['none']['correct'] / category_stats['none']['total'] * 100 if category_stats['none']['total'] > 0 else 0
    print(f"None Accuracy: {none_acc:.2f}% ({category_stats['none']['correct']}/{category_stats['none']['total']})")
    

    print(f"Combined Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    

    if USE_BALANCED_SUBSET:
        output_filename = "evaluation_results_balanced.txt"
    else:
        output_filename = "evaluation_results.txt"
    
    with open(output_filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Metadata
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Dataset: {test_jsonl_path}\n")
        f.write(f"Total Examples Evaluated: {total}\n\n")
        

        if USE_BALANCED_SUBSET:
            f.write("-" * 40 + "\n")
            f.write("SUBSET CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Balanced Subset: Enabled\n")
            f.write(f"None Percentage: {NONE_PERCENTAGE} {'(same as self-reference)' if NONE_PERCENTAGE == -1 else ''}\n")
            f.write(f"Random Seed: {RANDOM_SEED}\n\n")
            
            f.write("Original Dataset Distribution:\n")
            f.write(f"  Self-reference: {distribution_info['original']['self-reference']}\n")
            f.write(f"  Else: {distribution_info['original']['else']}\n")
            f.write(f"  None: {distribution_info['original']['none']}\n")
            f.write(f"  Total: {distribution_info['original']['total']}\n\n")
            
            f.write("Balanced Subset Distribution:\n")
            f.write(f"  Self-reference: {distribution_info['subset']['self-reference']}\n")
            f.write(f"  Else: {distribution_info['subset']['else']}\n")
            f.write(f"  None: {distribution_info['subset']['none']}\n")
            f.write(f"  Total: {distribution_info['subset']['total']}\n\n")
        else:
            f.write("Balanced Subset: Disabled (using full dataset)\n\n")
        

        f.write("-" * 40 + "\n")
        f.write("ACCURACY SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Self-reference Accuracy: {self_acc:.2f}% ({category_stats['self-reference']['correct']}/{category_stats['self-reference']['total']})\n")
        f.write(f"Else-reference Accuracy: {else_acc:.2f}% ({category_stats['else']['correct']}/{category_stats['else']['total']})\n")
        f.write(f"None Accuracy: {none_acc:.2f}% ({category_stats['none']['correct']}/{category_stats['none']['total']})\n")
        f.write(f"Combined Accuracy: {accuracy * 100:.2f}% ({correct}/{total})\n\n")
        

        f.write("-" * 40 + "\n")
        f.write("CATEGORY STATISTICS\n")
        f.write("-" * 40 + "\n")
        for cat, stats in category_stats.items():
            f.write(f"{cat.capitalize()}:\n")
            f.write(f"  Total: {stats['total']}\n")
            f.write(f"  Correct: {stats['correct']}\n")
            f.write(f"  Incorrect: {stats['total'] - stats['correct']}\n\n")
        

        f.write("=" * 80 + "\n")
        f.write(f"TRUE POSITIVES (up to 20 examples)\n")
        f.write("=" * 80 + "\n\n")
        for i, tp in enumerate(true_positives, 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"Prompt: {tp['prompt']}\n")
            f.write(f"Ground Truth: {tp['ground_truth']}\n")
            f.write(f"Generated: {tp['generated']}\n")
            f.write(f"Category: {tp['primary_category']}\n\n")
        
        if len(true_positives) == 0:
            f.write("No True Positives found.\n\n")
        

        f.write("=" * 80 + "\n")
        f.write(f"FALSE POSITIVES (up to 20 examples)\n")
        f.write("=" * 80 + "\n\n")
        for i, fp in enumerate(false_positives, 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"Prompt: {fp['prompt']}\n")
            f.write(f"Ground Truth: {fp['ground_truth']}\n")
            f.write(f"Generated: {fp['generated']}\n")
            f.write(f"GT Type: {fp['gt_type']}, Gen Type: {fp['gen_type']}\n")
            f.write(f"GT Mentions: {fp['gt_mentions']}\n")
            f.write(f"Gen Mentions: {fp['gen_mentions']}\n\n")
        
        if len(false_positives) == 0:
            f.write("No False Positives found.\n\n")
        

        f.write("=" * 80 + "\n")
        f.write(f"FALSE NEGATIVES (up to 20 examples)\n")
        f.write("=" * 80 + "\n\n")
        for i, fn in enumerate(false_negatives, 1):
            f.write(f"--- Example {i} ---\n")
            f.write(f"Prompt: {fn['prompt']}\n")
            f.write(f"Ground Truth: {fn['ground_truth']}\n")
            f.write(f"Generated: {fn['generated']}\n")
            f.write(f"GT Type: {fn['gt_type']}, Gen Type: {fn['gen_type']}\n")
            f.write(f"GT Mentions: {fn['gt_mentions']}\n")
            f.write(f"Gen Mentions: {fn['gen_mentions']}\n\n")
        
        if len(false_negatives) == 0:
            f.write("No False Negatives found.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("EXAMPLE COUNTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"True Positives collected: {len(true_positives)}\n")
        f.write(f"False Positives collected: {len(false_positives)}\n")
        f.write(f"False Negatives collected: {len(false_negatives)}\n")
    
    print(f"\nDetailed evaluation results saved to '{output_filename}'")
    print(f"True Positives collected: {len(true_positives)}")
    print(f"False Positives collected: {len(false_positives)}")
    print(f"False Negatives collected: {len(false_negatives)}")

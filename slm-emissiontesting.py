#%%

# This is the Notebook for our Carbonemissions test


#%%

from codecarbon import EmissionsTracker
import torch
import glob
import pandas as pd
import numpy as np
import re
from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTTrainer
from datasets import Dataset
import glob
from codecarbon import OfflineEmissionsTracker


#%%

import logging
tracker = OfflineEmissionsTracker(country_iso_code='AUT')

def find_csv_files(path, file_extension="*.csv"):
    try:
        files = glob.glob(f"{path}/{file_extension}")
        if not files:
            logging.warning(f"No files found in {path} with extension {file_extension}")
        return files
    except Exception as e:
        logging.error(f"Error finding files in {path}: {e}")
        return []

def read_csv_files(file_paths, column_name='Lyrics'):
    df_list = []
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            if column_name in df.columns:
                df_list.append(df)
            else:
                logging.warning(f"Column {column_name} not found in {file}")
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

def concatenate_lyrics(df, column_name='Lyrics'):
    if column_name in df.columns:
        try:
            return '\n'.join(str(lyric) for lyric in df[column_name])
        except Exception as e:
            logging.error(f"Failed to concatenate lyrics: {e}")
            return ""
    else:
        logging.warning(f"Column {column_name} not found in DataFrame")
        return ""

def load_and_concatenate_lyrics(path, file_extension="*.csv", column_name='Lyrics'):
    files = find_csv_files(path, file_extension)
    if not files:
        return ""
    df = read_csv_files(files, column_name)
    if df.empty:
        return ""
    return concatenate_lyrics(df, column_name)
path = 'datasets/'
lyrics = load_and_concatenate_lyrics(path)
print(lyrics[:200])

#%%

print(' '.join(sorted(set(lyrics))))

#%%

import re

def replace_characters(text, replacement_dict):
    return text.translate(str.maketrans(replacement_dict))

def remove_patterns(text, pattern_list):
    for pattern in pattern_list:
        text = re.sub(pattern, '', text)
    return text

def clean_lyrics(lyrics):
    replace_with_space = ['\u2005', '\u200b', '\u205f', '\xa0', '-']
    replace_letters = {'í':'i', 'é':'e', 'ï':'i', 'ó':'o', ';':',', '‘':'\'', '’':'\'', ':':',', 'е':'e'}
    remove_list = ['\)', '\(', '–','"','”', '"', '\[.*\]', '.*\|.*', '—']
    lyrics = replace_characters(lyrics, replace_letters)
    for string in replace_with_space:
        lyrics = lyrics.replace(string, ' ')
    lyrics = remove_patterns(lyrics, remove_list)
    return lyrics
cleaned_lyrics = clean_lyrics(lyrics)

#%%

print(''.join(sorted(set(cleaned_lyrics))))

#%%

cleaned_lyrics

#%%

from datasets import Dataset

def create_train_test_datasets(cleaned_lyrics, train_ratio=0.95, segment_length=500):
    split_point = int(len(cleaned_lyrics) * train_ratio)
    train_data = cleaned_lyrics[:split_point]
    test_data = cleaned_lyrics[split_point:]
    train_data_segments = [train_data[i:i + segment_length]
                           for i in range(0, len(train_data), segment_length)]
    train_dataset = Dataset.from_dict({'text': train_data_segments})
    return train_dataset, test_data
train_dataset, test_data = create_train_test_datasets(cleaned_lyrics)

#%%

print(len(train_dataset))

#%%

train_dataset

# Check the dataset structure
# Should output below
"""
Dataset({
    features: ['text'],
    num_rows: 557
})
"""

#%%

from pprint import pprint
pprint(train_dataset[0])

#%%

def load_quantized_model(model_identifier: str, compute_dtype: torch.dtype) -> AutoModelForCausalLM:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model =  AutoModelForCausalLM.from_pretrained(
        model_identifier,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return model

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = load_quantized_model(model_name, torch.bfloat16)

#%%

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token

#%%

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_lyrics(query, model):
    encoding = tokenizer(query, return_tensors="pt").to(device)
    generation_config = GenerationConfig(max_new_tokens=250, pad_token_id = tokenizer.eos_token_id,repetition_penalty=1.3, eos_token_id = tokenizer.eos_token_id)

    outputs = model.generate(input_ids=encoding.input_ids, generation_config=generation_config)

    text_output = tokenizer.decode(outputs[0],skip_special_tokens=True)
    print('INPUT\n', query, '\n\nOUTPUT\n', text_output[len(query):])

generate_lyrics(test_data[200:700], model)

#%%

model = prepare_model_for_kbit_training(model)
lora_alpha = 32
lora_dropout = 0.05
lora_rank = 32
lora_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_rank,
    bias="none",
    task_type="CAUSAL_LM")
peft_model = get_peft_model(model, lora_config)

#%%

output_dir = "hf-username/tinylama_taylor_swift"
per_device_train_batch_size = 3
gradient_accumulation_steps = 2
optim = "paged_adamw_32bit"
save_strategy="steps"
save_steps = 10
logging_steps = 10
learning_rate = 2e-3
max_grad_norm = 0.3
max_steps = 200
warmup_ratio = 0.03
lr_scheduler_type = "cosine"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    report_to='none'
)

#%%

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    max_seq_length=500,
    dataset_text_field='text',
    tokenizer=tokenizer,
    args=training_arguments
)
peft_model.config.use_cache = False

#%%

tracker.start()
trainer.train()
tracker.stop()

#%%



import sys

import torch
import pandas as pd
from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTTrainer

import glob
from codecarbon import OfflineEmissionsTracker

import logging
import warnings

# Suppress the FutureWarning from huggingface_hub
warnings.filterwarnings("ignore",
                        message="`resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.")

# Suppress the UserWarning from torch.utils.checkpoint
warnings.filterwarnings("ignore",
                        message="torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.")

tracker = OfflineEmissionsTracker(country_iso_code='AUT')

logger = logging.getLogger("codecarbon")
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])

    # Define a log formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s"
)

##Code for logging taken from: https://github.com/mlco2/codecarbon/blob/master/examples/logging_to_file.py
# Create file handler which logs debug messages
fh = logging.FileHandler("results/Logging Results/tinyllama/codecarbon_sample.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(formatter)
consoleHandler.setLevel(logging.WARNING)
logger.addHandler(consoleHandler)


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

print(' '.join(sorted(set(lyrics))))

import re


def replace_characters(text, replacement_dict):
    return text.translate(str.maketrans(replacement_dict))


def remove_patterns(text, pattern_list):
    for pattern in pattern_list:
        text = re.sub(pattern, '', text)
    return text


def clean_lyrics(lyrics):
    replace_with_space = ['\u2005', '\u200b', '\u205f', '\xa0', '-']
    replace_letters = {'í': 'i', 'é': 'e', 'ï': 'i', 'ó': 'o', ';': ',', '‘': '\'', '’': '\'', ':': ',', 'е': 'e'}
    remove_list = ['\)', '\(', '–', '"', '”', '"', '\[.*\]', '.*\|.*', '—']
    lyrics = replace_characters(lyrics, replace_letters)
    for string in replace_with_space:
        lyrics = lyrics.replace(string, ' ')
    lyrics = remove_patterns(lyrics, remove_list)
    return lyrics


cleaned_lyrics = clean_lyrics(lyrics)
# %%
print(''.join(sorted(set(cleaned_lyrics))))
# %%
cleaned_lyrics
# %%
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
# %%
print(len(train_dataset))
# %%
train_dataset

# Check the dataset structure
# Should output below
"""
Dataset({
    features: ['text'],
    num_rows: 557
})
"""

from pprint import pprint

pprint(train_dataset[0])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_lyrics(query, model):
    encoding = tokenizer(query, return_tensors="pt").to(device)
    generation_config = GenerationConfig(max_new_tokens=500, pad_token_id=tokenizer.eos_token_id,
                                         repetition_penalty=1.3, eos_token_id=tokenizer.eos_token_id)
    num_samples = 10
    for k in range(num_samples):
        outputs = model.generate(input_ids=encoding.input_ids, generation_config=generation_config)
        text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('INPUT\n', query, '\n\nOUTPUT\n', text_output[len(query):])
        print("---------------------")


checkpoint_path = "hf-username/tinylama_taylor_swift/checkpoint-30000"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

tracker.start()
generate_lyrics(test_data[200:700], model)
tracker.stop()

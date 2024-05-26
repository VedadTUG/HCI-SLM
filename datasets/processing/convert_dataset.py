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
import tqdm as tqdm

import logging
import csv

# Define the input and output file paths
input_file = "../shakespeare.csv"
output_file = '../shakespeare_out.csv'

# Open the input file for reading
with open(input_file, mode='r', newline='') as infile:
    reader = csv.reader(infile)

    # Open the output file for writing
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Iterate over each row in the input file
        for row in reader:
            # Add two empty entries at the beginning of each row
            new_row = [' ', ' ', *row]
            # Write the new row to the output file
            writer.writerow(new_row)

print(f"Modified data has been saved to {output_file}")
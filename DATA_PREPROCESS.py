"""
DATA IMPORT
"""

import pandas as pd
import os

# Make OS-paths
current_directory = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(current_directory, 'it/train.tsv')
DEV_PATH = os.path.join(current_directory, 'it/dev.tsv')
TEST_PATH = os.path.join(current_directory, 'it/test.tsv')
CLIPS_PATH = os.path.join(current_directory, 'it/clips')
PREPROCESSED_DIR = os.path.join(current_directory, 'data_preprocessed')

# Load metadata into pandas DataFrame
train_df = pd.read_csv(TRAIN_PATH, sep='\t')
dev_df = pd.read_csv(DEV_PATH, sep='\t')
test_df = pd.read_csv(TEST_PATH, sep='\t')

_ACCENTS_IT = [
    1e4, # max 10000 samples per accent
    "Emiliano", # 151
    "Meridionale", # 193
    "Veneto", # 1508
    "Tendente al siculo, ma non marcato", # 2175
    "Basilicata,trentino", # 2297
]

def process_tsv(file_path, accents, clips_path):
    data = pd.read_csv(file_path, sep='\t')
    data['file_name'] = data['path'] + '.mp3'
    data['file_path'] = data['file_name'].apply(lambda x: os.path.join(clips_path, x))
    data = data[data['accent'].isin(accents)]  # Filter rows where accent is in the accents list
    return data[['file_name', 'file_path', 'accent']]

def save_to_csv(data, output_path):
    data.to_csv(output_path, index=False)

# Main script logic
if not os.path.exists(PREPROCESSED_DIR):
    os.makedirs(PREPROCESSED_DIR)

train_data = process_tsv(TRAIN_PATH, _ACCENTS_IT, CLIPS_PATH)
dev_data = process_tsv(DEV_PATH, _ACCENTS_IT, CLIPS_PATH)
test_data = process_tsv(TEST_PATH, _ACCENTS_IT, CLIPS_PATH)

save_to_csv(train_data, os.path.join(PREPROCESSED_DIR, 'train_processed.csv'))
save_to_csv(dev_data, os.path.join(PREPROCESSED_DIR, 'dev_processed.csv'))
save_to_csv(test_data, os.path.join(PREPROCESSED_DIR, 'test_processed.csv'))
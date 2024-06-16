import os
import argparse
import csv
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def prepare_cv_from_local():
    """ Function to prepare the datasets from local files in <input_folder> """

    current_directory = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(current_directory, 'it')
    output_folder = os.path.join(current_directory, 'output_folder')

    # create the output folder: in case it is not present
    os.makedirs(output_folder, exist_ok=True)
    
    # List of files to process
    files = ['dev.tsv', 'other.tsv', 'test.tsv', 'train.tsv', 'valideted.tsv']
    
    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping...")
            continue
        
        df = pd.read_csv(file_path, sep='\t')
        
        # Filter out samples without accent if 'accent' column exists
        if 'accent' in df.columns:
            df = df[df['accent'] != '']
        
        # Processing the dataframe and adding an index column
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'idx'}, inplace=True)
        
        # Selecting and reordering columns if necessary
        columns = ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent']
        if not all(col in df.columns for col in columns):
            print(f"File {file_name} does not have the required columns. Skipping...")
            continue
        
        df = df[columns]
        
        # Write the processed dataframe to a new TSV file
        output_file = os.path.join(output_folder, file_name)
        df.to_csv(output_file, sep='\t', index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Processed and saved {file_name} to {output_file}")

prepare_cv_from_local()
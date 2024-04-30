""" converts parquet file into .csv and .txt files for training """

import os
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

input_directory = '../pre-train/parquet files'
output_directory = '../pre-train/output'

if not os.path.exists(output_directory):
  os.makedirs(output_directory)

parquet_files = [file for file in os.listdir(input_directory) if file.endswith('.parquet')]

for file_name in parquet_files:
  input_path = os.path.join(input_directory, file_name)
  base_name = os.path.splitext(file_name)[0]

  # Convert to CSV
  csv_output_path = os.path.join(output_directory, base_name + '.csv')
  df = pd.read_parquet(input_path)
  df.to_csv(csv_output_path, sep=',', index=False)
  print(f"CSV Conversion complete: {file_name} -> {csv_output_path}")

  # Convert to TXT
  txt_output_path = os.path.join(output_directory, base_name + '.txt')
  df.to_csv(txt_output_path, sep='\t', index=False)
  print(f"TXT Conversion complete: {file_name} -> {txt_output_path}")

print("All Parquet files converted to CSV and TXT files.")
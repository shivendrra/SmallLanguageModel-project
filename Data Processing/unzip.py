""" 
this code is for unzipping and extracting files
"""

import os
import gzip
import shutil

os.chdir("D:/Machine Learning/SLM-Project/")

def unzip_all_files(input_directory, output_directory):
  os.makedirs(output_directory, exist_ok=True)

  # List all files in the input directory
  files = os.listdir(input_directory)

    # Iterate over each file in the input directory
  for file_name in files:
    input_path = os.path.join(input_directory, file_name)
    output_path = os.path.join(output_directory, os.path.splitext(file_name)[0])

        # Check if the file is a GZip file
    if file_name.endswith('.gz'):
      print(f"Unzipping: {file_name}")

      # Open the GZip file and decompress the data
      with gzip.open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)

          print(f"Unzipping complete: {output_path}")
    else:
      print(f"Skipping non-GZip file: {file_name}")

if __name__ == "__main__":
  input_directory = "Data/zipped files"
  output_directory = "Data/zipped files"

  unzip_all_files(input_directory, output_directory)

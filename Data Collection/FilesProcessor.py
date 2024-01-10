import os
os.chdir("D:/Machine Learning/SLM-Project/")

def combine_text_files(input_directory, output_file):
  files = os.listdir(input_directory)

  with open(output_file, 'a', encoding='utf-8') as output_file:
    for file_name in files:
      input_path = os.path.join(input_directory, file_name)

      if file_name.endswith('.txt') and os.path.isfile(input_path):
        print(f"Reading: {file_name}")
        with open(input_path, 'r', encoding='utf-8') as input_file:
          output_file.write(input_file.read())
          output_file.write('\n')

        print(f"Reading complete: {file_name}")
      else:
        print(f"Skipping non-text file: {file_name}")

if __name__ == "__main__":
  input_directory = "Data/txt files"
  output_file = "Data/txt files/big_data.txt"
  combine_text_files(input_directory, output_file)
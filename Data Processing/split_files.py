import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

def split_file(input_file, output_dir, num_files):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        lines_per_file = total_lines // num_files
        f.seek(0)
        
        for i in range(num_files):
            output_file = os.path.join(output_dir, f'chunk_0{i+1}.txt')
            with open(output_file, 'w', encoding='utf-8') as fw:
                lines_written = 0
                while lines_written < lines_per_file:
                    line = f.readline()
                    if not line:
                        break
                    fw.write(line)
                    lines_written += 1

    print(f"File split completed into {num_files} files in '{output_dir}' directory.")

if __name__ == "__main__":
    input_file_path = '../pre-train/parquet files/consolidated_6b.txt'
    output_directory = '../datasets'
    num_output_files = 4
    
    split_file(input_file_path, output_directory, num_output_files)
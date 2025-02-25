import time
import hashlib
import os
import pandas as pd

def compute_hash(file_path, algorithm='sha256'):
    """Compute the hash of a file using the specified algorithm."""
    hash_obj = hashlib.new(algorithm)
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error: {str(e)}"

def hash_files_from_excel(excel_path, file_column="Filename", hash_column="Hash", base_dir=""):
    """
    Read filenames from an Excel file, compute their hashes, and update the Excel file.
    
    Parameters:
    - excel_path: Path to the Excel file
    - file_column: Column name containing the filenames
    - hash_column: Column name to store the hashes
    - base_dir: Base directory for the files (if filenames in Excel don't include the full path)
    """
    start_time = time.perf_counter()
    
    # Read the Excel file
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    # Add hash column if it doesn't exist
    if hash_column not in df.columns:
        df[hash_column] = ""
    
    # Process each file
    for index, row in df.iterrows():
        if file_column not in df.columns:
            print(f"Column '{file_column}' not found in Excel file.")
            return
            
        filename = row[file_column]
        if not isinstance(filename, str):
            df.at[index, hash_column] = "Invalid filename"
            continue
            
        # Combine base directory with filename if base_dir is provided
        file_path = os.path.join(base_dir, filename) if base_dir else filename
        
        # Compute hash
        file_hash = compute_hash(file_path)
        df.at[index, hash_column] = file_hash
        print(f"Processed: {filename} -> {file_hash[:10]}...")
    
    # Save the updated Excel file
    try:
        df.to_excel(excel_path, index=False)
        print(f"Updated Excel file saved: {excel_path}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")
    
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    excel_file = "files_to_hash.xlsx"  # Path to your Excel file
    hash_files_from_excel(
        excel_file,
        file_column="Filename",  # Column containing filenames 
        hash_column="FileHash",  # Column to store hashes
        base_dir=""  # Optional base directory for files
    )

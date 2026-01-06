
import scipy.io
import numpy as np
import os

def print_structure(data, indent=0):
    indent_str = " " * indent
    if isinstance(data, dict):
        for key in data:
            if key.startswith('__'): continue
            value = data[key]
            print(f"{indent_str}{key}: {type(value)}")
            if isinstance(value, (dict, np.ndarray, np.void)):
                print_structure(value, indent + 2)
    elif isinstance(data, np.ndarray):
        print(f"{indent_str}Shape: {data.shape}, Dtype: {data.dtype}")
        if data.size < 10:
            print(f"{indent_str}Values: {data}")
        elif data.dtype.names: # Structured array
            print(f"{indent_str}Fields: {data.dtype.names}")
            # Recursively print first element if possible
            if data.size > 0:
                 print_structure(data[0], indent + 2)
    elif isinstance(data, np.void): # Mat file struct
        print(f"{indent_str}Struct Fields: {data.dtype.names}")
        for name in data.dtype.names:
            print(f"{indent_str}Field: {name}")
            print_structure(data[name], indent + 2)

def main():
    file_path = 'data/DREAMER.mat'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    try:
        mat = scipy.io.loadmat(file_path)
        print("Keys in mat file:", mat.keys())
        
        if 'DREAMER' in mat:
            print("\nlnspecting 'DREAMER' struct:")
            # DREAMER is usually a 1x1 struct
            dreamer = mat['DREAMER']
            print(f"Shape: {dreamer.shape}")
            # It's likely a structured array of shape (1,1)
            if dreamer.shape == (1,1):
                data = dreamer[0,0]
                print("Fields:", data.dtype.names)
                for field in data.dtype.names:
                    print(f"\n--- Field: {field} ---")
                    val = data[field]
                    print(f"Shape: {val.shape}")
                    # Peek inside
                    if val.size > 0:
                        sub = val[0,0] if val.ndim > 1 else val[0]
                        if isinstance(sub, np.void) or isinstance(sub, np.ndarray):
                           pass # Too deep, let's stop here or be selective
                        
    except Exception as e:
        print(f"Error loading mat file: {e}")

if __name__ == "__main__":
    main()

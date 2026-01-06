
import sys
import os
print("Python executable:", sys.executable)
print("Python version:", sys.version)
try:
    import scipy
    print("SciPy version:", scipy.__version__)
    import scipy.io
    print("SciPy IO module loaded")
except ImportError as e:
    print("Error importing SciPy:", e)

file_path = 'data/DREAMER.mat'
if os.path.exists(file_path):
    print(f"File {file_path} exists, size: {os.path.getsize(file_path)} bytes")
else:
    print(f"File {file_path} NOT found")

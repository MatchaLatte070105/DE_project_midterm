import numpy as np

data = np.load("bungee_jump.npz", allow_pickle=True)

print("Keys in the file:", data.files)
for key in data.files:
    arr = data[key]
    print(f"  {key}: shape = {getattr(arr, 'shape', None)}, dtype = {getattr(arr, 'dtype', type(arr))}")
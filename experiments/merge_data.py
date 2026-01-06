import numpy as np

files = [
    "ppRun1_diffusion_data.npz",
    "ppRun2_diffusion_data.npz",
    "ppeRun1_diffusion_data.npz",
]

data = [np.load(f) for f in files]

merged = {}
for key in data[0].files:
    merged[key] = np.concatenate([d[key] for d in data], axis=0)

np.savez("pp_diffusion_data_merged.npz", **merged)
print("Merged data saved to diffusion_data_merged.npz")
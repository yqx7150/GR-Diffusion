from glob import glob
import os
import numpy as np

root_path = "output/p1_D20_v2"

if __name__ == "__main__":
    parts = sorted(glob(f"{root_path}/*/*"))
    
    whole_body = []
    for part in parts:
        print(part)
        arr = np.load(part)
        arr = np.pad(arr, pad_width=((0, 0), (52, 52), (52, 52)))
        whole_body.append(arr)
    whole_body.append(np.zeros((1, 360, 360)))
    whole_body = np.concatenate(whole_body, axis=0)
    print(whole_body.shape)

    C, H, W = whole_body.shape
    np.save(f"{root_path}/DGR_whole_body_{C}x{H}x{W}.npy", whole_body)
        

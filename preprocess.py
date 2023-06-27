import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from joblib import Parallel, delayed
import argparse



def func(i):
    
    path = f"{fragpath}/{i:02}.tif"

    tmp = cv2.imread(path)
    tmp = np.clip(tmp,0,255)
    tmp = tmp.astype("uint8")
    filename = path.split("/")[-1]
    cv2.imwrite(f"{opt.inputpath}/{frag}/{filename}",tmp)
    
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputpath", type=str, default='.')
    opt = parser.parse_args()

    frags = os.listdir(opt.inputpath)
    frags.sort()
    frags

    for frag in frags:
        print(f"frag {frag} start")
        fragpath = f"{opt.inputpath}/{frag}/surface_volume"
        _ = Parallel(n_jobs = -1, verbose = 1)(delayed(func)(i) for i in range(65))

    print("preprocess done")

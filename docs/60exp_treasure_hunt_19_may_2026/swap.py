import struct
import glob
import os
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timezone

# first perform the byte swap and save swapped as additional file
inpath = ''
files_d = glob.glob(inpath+'*.bin')
for file in files_d:
    outfile = file[:-4]+'_swapped.bin'
    data = np.fromfile(file, dtype=np.uint64)
    data.byteswap(inplace=True)
    data.tofile(outfile)
    print(f"Swapped: {file} -> {outfile}")


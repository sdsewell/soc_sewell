import struct
import glob
import os
import h5py
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timezone

# first perform the byte swap and save swapped as additional file
inpath = ''
files_d = glob.glob(inpath+'*.bin')
for file in files_d:
    outfile = file[:-4]+'_swapped.bin'
    subprocess.run(['./swap.sh', file, outfile])


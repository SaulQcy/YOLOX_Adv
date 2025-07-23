import torch
import numpy as np
import random
import time
import os

def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import os
import shutil

def rm_if_exist(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Folder '{folder}' has been removed.")
    else:
        print(f"Folder '{folder}' does not exist, no action taken.")
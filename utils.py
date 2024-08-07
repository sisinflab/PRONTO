import random
import torch
import numpy as np
from config import RANDOM_SEED
import torch.functional as F

def enable_determinism(random_seed=RANDOM_SEED):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
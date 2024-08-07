import torch
import os


RANDOM_SEED = 123
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

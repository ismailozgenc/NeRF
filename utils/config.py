# config.py
import torch 
DATA_DIR     = "data/south-building/sparse"
IMG_DIR      = "data/south-building/images"
N_ITERS      = 200000
LR           = 4e-4
BATCH_RAYS   = 1024
N_SAMPLES    = 64 # can be doubled IG 
FREQ_POS     = 10
FREQ_DIR     = 4
LOG_INTERVAL = 100
CKPT_DIR     = "checkpoints"
N_IMPORTANCE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEAR = 2
FAR = 6
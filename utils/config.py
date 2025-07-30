# config.py
DATA_DIR     = "data/south-building/sparse"
IMG_DIR      = "data/south-building/images"
N_ITERS      = 200000
LR           = 5e-4
BATCH_RAYS   = 1024
N_SAMPLES    = 64
NEAR, FAR    = 2.0, 6.0
FREQ_POS     = 10
FREQ_DIR     = 4
LOG_INTERVAL = 100
CKPT_DIR     = "checkpoints"
N_IMPORTANCE = 128
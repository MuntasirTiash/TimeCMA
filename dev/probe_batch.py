# dev/probe_batch.py
import torch
from train import parse_args, load_data

args = parse_args()
args.data_path   = "ETTh1"
args.seq_len     = 96
args.pred_len    = 96
args.batch_size  = 8
args.num_workers = 0  # deterministic, no multiprocessing

train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)

xb, yb, xmb, ymb, emb = next(iter(train_loader))
print("x:", xb.shape, "y:", yb.shape)
print("x_mark:", xmb.shape, "y_mark:", ymb.shape)
print("embeddings:", emb.shape)
print("dtype:", xb.dtype, yb.dtype, "device:", xb.device)
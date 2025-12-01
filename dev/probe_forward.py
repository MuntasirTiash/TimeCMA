# dev/probe_forward.py
import torch
from train import parse_args, load_data, trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
args.data_path, args.seq_len, args.pred_len = "ETTh1", 96, 96
args.batch_size, args.num_workers = 8, 0

_, _, _, _, _, test_loader, scaler = load_data(args)
xb, yb, xmb, ymb, emb = next(iter(test_loader))
xb, yb, xmb, ymb, emb = xb.to(device), yb.to(device), xmb.to(device), ymb.to(device), emb.to(device)

eng = trainer(scaler, args.channel, args.num_nodes, args.seq_len, args.pred_len,
              args.dropout_n, args.d_llm, args.e_layer, args.d_layer, args.head,
              args.learning_rate, args.weight_decay, device, args.epochs)
model = eng.model.to(device).eval()

# Optional: forward hooks to print shapes per module
def hook(name):
    def fn(mod, inp, out):
        si = [tuple(getattr(t, 'shape', ())) for t in (inp if isinstance(inp, (list,tuple)) else [inp])]
        so = tuple(getattr(out, 'shape', ())) if hasattr(out, 'shape') else out
        print(f"[{name}] in:{si} out:{so}")
    return fn

for name, m in list(model.named_modules())[:10]:  # print a few top modules; expand as needed
    m.register_forward_hook(hook(name))

with torch.no_grad():
    yhat = model(xb, xmb, emb)
print("yhat:", yhat.shape, "target:", yb.shape)
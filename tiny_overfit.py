"""tiny_overfit.py -- MANDATORY sanity check. Run BEFORE any further full training.
Trains on 10 image-caption pairs. Uses your unmodified modules."""
import torch, pandas as pd, torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset import ROCODataset
from vocab import Vocab
from caption_model import SwinCaptioningModel
from train import build_param_groups

ROOT="/Users/nagashiva/Downloads/rocov2"; N=10; MAX_LEN=40
DEVICE=torch.device("mps" if torch.backends.mps.is_available() else "cpu")

df=pd.read_csv(f"{ROOT}/train_captions.csv").iloc[:N]
vocab=Vocab(df["Caption"].astype(str).tolist(), min_freq=1)   # min_freq=1: no <unk> on 10 samples
ds=ROCODataset(f"{ROOT}/train_captions.csv", f"{ROOT}/train_images/train", vocab, MAX_LEN, max_samples=N)
dl=DataLoader(ds,batch_size=N,shuffle=False)

model=SwinCaptioningModel(vocab_size=len(vocab),max_len=MAX_LEN).to(DEVICE)
opt=torch.optim.AdamW(build_param_groups(model,0.0),lr=1e-4)     # NO weight decay, LOW lr
crit=torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)        # NO label smoothing
imgs,caps=next(iter(dl)); imgs,caps=imgs.to(DEVICE),caps.to(DEVICE)

for step in range(400):
    model.train()
    logits=model(imgs,caps); loss=crit(logits.reshape(-1,logits.size(-1)),caps[:,1:].reshape(-1))
    opt.zero_grad(); loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1e9)     # measure, don't clip
    opt.step()
    if step%25==0:
        enc=sum(p.grad.norm()**2 for n,p in model.named_parameters() if n.startswith("encoder") and p.grad is not None)**0.5
        dec=sum(p.grad.norm()**2 for n,p in model.named_parameters() if n.startswith("decoder") and p.grad is not None)**0.5
        print(f"step {step:4d}  loss {loss.item():.4f}  |grad| total {gn:.3f}  enc {enc:.4f}  dec {dec:.4f}  ratio {enc/dec:.4f}")

model.eval()
with torch.no_grad():
    for i in range(N):
        ids=model.generate(imgs[i:i+1],vocab,max_len=MAX_LEN,device=DEVICE)
        print(f"\nGT : {df.iloc[i]['Caption'][:90]}\nGEN: {vocab.decode(ids[0].tolist())}")

"""
Inference: load a trained checkpoint and generate a caption for one image.

Usage:
    python generate.py /path/to/some_image.jpg /path/to/rocov2/swin_caption_epoch10.pt
"""

import sys
import torch
from PIL import Image
import torchvision.transforms as T

from vocab import Vocab
from caption_model import SwinCaptioningModel

MAX_LEN = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_vocab_from_checkpoint(ckpt):
    v = Vocab.__new__(Vocab)   # bypass __init__, restore state directly
    v.itos = ckpt["vocab_itos"]
    v.stoi = {w: i for i, w in enumerate(v.itos)}
    v.pad_token, v.sos_token, v.eos_token, v.unk_token = "<pad>", "<sos>", "<eos>", "<unk>"
    v.pad_id = v.stoi[v.pad_token]
    v.sos_id = v.stoi[v.sos_token]
    v.eos_id = v.stoi[v.eos_token]
    v.unk_id = v.stoi[v.unk_token]
    return v


def main(image_path, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    vocab = load_vocab_from_checkpoint(ckpt)

    model = SwinCaptioningModel(vocab_size=len(vocab), max_len=MAX_LEN).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_t = transform(image).unsqueeze(0).to(DEVICE)

    ids = model.generate(image_t, vocab, max_len=MAX_LEN, device=DEVICE)
    caption = vocab.decode(ids[0].tolist())
    print("Generated caption:", caption)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

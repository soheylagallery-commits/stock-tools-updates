# CLIP ViT-B/32 text embeddings (same model as the app's image encoder), L2-normalised, float16
import numpy as np, torch, os, hashlib
from transformers import CLIPModel, CLIPTokenizer
src = "tools/vocab/phrases.txt"; out = "tools/vocab/out"; os.makedirs(out, exist_ok=True)
P = [l.strip() for l in open(src, encoding="utf-8") if l.strip()]
m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(); tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
E = []
with torch.no_grad():
    for i in range(0, len(P), 256):
        b = tok(["a photo of " + p for p in P[i:i+256]], padding=True, truncation=True, max_length=77, return_tensors="pt")
        e = m.get_text_features(**b); e = e / e.norm(dim=-1, keepdim=True); E.append(e.numpy())
E = np.concatenate(E).astype(np.float16)
E.tofile(out + "/emb.f16")
open(out + "/phrases.txt", "w", encoding="utf-8").write("\n".join(P))
open(out + "/sha.txt", "w").write(hashlib.sha256(open(src, "rb").read()).hexdigest())
print(len(P), E.shape)

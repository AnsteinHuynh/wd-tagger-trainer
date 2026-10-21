import json, os, time
import torch

CN = r"C:\MyApps\ComfyUIRoot\convnextv2_huge.dbv4-full"
DINO = r"I:\Models\dinov3-vitb16-pretrain-lvd1689m"

t0 = time.time()
import timm
from safetensors.torch import load_file

m = timm.create_model("convnextv2_huge", pretrained=False, num_classes=12476,
                      drop_path_rate=0.4, act_layer="gelu_tanh", drop_rate=0.2)
sd = load_file(os.path.join(CN, "model.safetensors"))
missing, unexpected = m.load_state_dict(sd, strict=False)
print("timms load ok; missing:", len(missing), missing[:5], "unexpected:", len(unexpected), unexpected[:5])
print("params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
m = m.cuda().to(torch.bfloat16).eval()
x = torch.randn(8, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    t = time.time(); y = m(x); torch.cuda.synchronize(); print("baseline fwd(8@512) bf16: %.3fs" % (time.time() - t), y.shape, y.dtype)
del m, x, y
torch.cuda.empty_cache()
print("vram after baseline free: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

# ---- DINOv3 ----
from transformers import DINOv3ViTModel
dm = DINOv3ViTModel.from_pretrained(DINO, dtype=torch.bfloat16).cuda()
print("dinov3 params:", sum(p.numel() for p in dm.parameters()) / 1e6, "M")
x = torch.randn(8, 3, 512, 512, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    t = time.time(); out = dm(pixel_values=x); torch.cuda.synchronize()
    print("dinov3 fwd(8@512) bf16: %.3fs" % (time.time() - t))
    print(type(out), [k for k in out.keys()] if hasattr(out, "keys") else "")
    print("last_hidden_state:", out.last_hidden_state.shape)
    print("pooler:", None if out.pooler_output is None else out.pooler_output.shape)

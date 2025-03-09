import torch
from modelscope import AutoTokenizer, AutoModel
path = "OpenGVLab/InternVL2-40B"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
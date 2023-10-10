import torch

from src.common import decode_func, encode_func


with open("./data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

encode = encode_func(chars)
decode = decode_func(chars)


print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

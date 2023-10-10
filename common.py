# create a mapping from characters to integers
stoi_func = lambda chars: {ch: i for i, ch in enumerate(chars)}
itos_func = lambda chars: {i: ch for i, ch in enumerate(chars)}

encode_func = lambda chars: lambda s: [
    stoi_func(chars)[c] for c in s
]  # encoder: take a string, output a list of integers

decode_func = lambda chars: lambda l: "".join(
    [itos_func(chars)[i] for i in l]
)  # decoder: take a list of integers, output a string
def stoi_func(chars):
    return {ch: i for i, ch in enumerate(chars)}


def itos_func(chars):
    return {i: ch for i, ch in enumerate(chars)}


def encode_func(chars):
    return lambda s: [stoi_func(chars)[c] for c in s]


def decode_func(chars):
    return lambda s: "".join(
        [itos_func(chars)[i] for i in s]
    )  # decoder: take a list of integers, output a string

with open("./data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("char length", len(text))

# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)
import tiktoken

def main():
    print("Tokenizing a URL using GPT-4's tokenizer (cl100k_base):\n")
# Load GPT-4's tokenizer (cl100k_base)
    enc = tiktoken.get_encoding("cl100k_base")

    url = "https://en.wikipedia.org/wiki/Hackers_(film)"

# Encode into token IDs
    token_ids = enc.encode(url)

# Decode back into string pieces for readability
    tokens = [enc.decode([t]) for t in token_ids]

    print("URL:", url)
    print("\nTokens:", tokens)
    print("\nhttps://", token_ids)
    print("\nToken count:", len(token_ids))


if __name__ == "__main__":
    main()


from .basic_examples import tokenizer as simple_tokenizer
from .basic_examples import tokenizer_utils as tu

import tiktoken

if __name__ == "__main__":
    # raw_txt = tu.read_txt_into_str("/BuildingLLMFromScratch/text_sources/the-verdict.txt")
    # vocab = tu.create_vocabulary(tu.split_txt_into_words(raw_txt))
    # tokenizer = simple_tokenizer.SimpleTokenizer(vocab)

    tokenizer = tiktoken.get_encoding("gpt2")

    text = "Hello, do you like tea? <|endoftext|> In the sunlit " \
    "terraces of someunknownPlace."
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(ids)
    strings = tokenizer.decode(ids)
    print(strings)
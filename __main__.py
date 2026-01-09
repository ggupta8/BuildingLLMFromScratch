# from basic_examples import tokenizer as simple_tokenizer
from basic_examples import tokenizer_utils as tu

# Byte Pair Encoding
import tiktoken

if __name__ == "__main__":
    # raw_txt = tu.read_txt_into_str("/BuildingLLMFromScratch/text_sources/the-verdict.txt")
    # vocab = tu.create_vocabulary(tu.split_txt_into_words(raw_txt))
    # tokenizer = simple_tokenizer.SimpleTokenizer(vocab)

    tokenizer = tiktoken.get_encoding("gpt2")

    text = tu.read_txt_into_str("/text_sources/the-verdict.txt")
    encoded_txt = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_sample = encoded_txt[50:]
    print(len(encoded_txt))

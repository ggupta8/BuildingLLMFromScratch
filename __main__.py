from basic_examples import dataloader_utils as du
# from basic_examples import tokenizer as simple_tokenizer
from basic_examples import tokenizer_utils as tu

if __name__ == "__main__":
    # raw_txt = tu.read_txt_into_str("/text_sources/the-verdict.txt")
    # vocab = tu.create_vocabulary(tu.split_txt_into_words(raw_txt))
    # tokenizer = simple_tokenizer.SimpleTokenizer(vocab)

    text = tu.read_txt_into_str("/text_sources/the-verdict.txt")
    dataloader = du.create_dataloader(text, batch_size=1, max_len=4,
                                      stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

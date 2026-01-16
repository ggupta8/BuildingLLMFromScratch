from basic_examples import dataloader_utils as du
# from basic_examples import tokenizer as simple_tokenizer
from basic_examples import tokenizer_utils as tu

if __name__ == "__main__":
    # raw_txt = tu.read_txt_into_str("/text_sources/the-verdict.txt")
    # vocab = tu.create_vocabulary(tu.split_txt_into_words(raw_txt))
    # tokenizer = simple_tokenizer.SimpleTokenizer(vocab)

    text = tu.read_txt_into_str("/text_sources/the-verdict.txt")
    dataloader = du.create_dataloader(text, batch_size=8, max_len=4,
                                      stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    input_embeddings = du.create_embeddings(256, inputs)
    print(input_embeddings.shape)

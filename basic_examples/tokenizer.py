import re
from . import tokenizer_utils

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    """
    Processes input text into token IDs.
    """
    def encode(self, text):
        split_txt = tokenizer_utils.split_txt_into_words(text)
        ids = [self.str_to_int[s] for s in split_txt]
        return ids
    
    """
    Converts token IDs back into text.
    """
    def decode(self, token_id_map):
        text = " ".join([self.int_to_str[i] for i in token_id_map])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

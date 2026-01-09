import re
from . import tokenizer_utils

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        """Processes input text into token IDs."""
        split_txt = tokenizer_utils.split_txt_into_words(text)
        # Replace unknown words with "unk".
        split_txt = [item if item in self.str_to_int
                     else "<|unk|>" for item in split_txt]
        ids = [self.str_to_int[word] for word in split_txt]
        return ids
    
    def decode(self, token_id_map):
        """Converts token IDs back into text."""
        text = " ".join([self.int_to_str[i] for i in token_id_map])
        # Replaces spaces before punctuations.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

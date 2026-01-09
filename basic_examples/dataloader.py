import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, txt, tokenizer, max_len, stride):
        """Creates list of input and target tensors from a text dataset.

        @param: txt Text dataset.
        @param: tokenizer Tokenizer used to encode dataset.
        @param: max_len Maximum length of input context words.
        @param: stride Number to shift input position (determines if overlapping).
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_len, stride):
            input_chunk = token_ids[i: i + max_len]
            target_chunk = token_ids[i + 1: i + max_len + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

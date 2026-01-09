from .dataloader import SimpleDataset
import tiktoken     # Byte Pair Encoding
from torch.utils.data import DataLoader

def create_dataloader(txt, batch_size=4, max_len=256,
                             stride=128, shuffle=True, drop_last=True,
                             num_workers=0):
        """Creates list of input and target tensors from a text dataset.

        @param: txt Text dataset.
        @param: batch_size Number of tensors in the input - target lists.
        @param: max_len Maximum length of input context words.
        @param: stride Number to shift input position (determines if overlapping).
        @param: shuffle Randomly permutes dataset indices. Prevents overfitting.
        @param: drop_last Whether to drop the last batch if < batch_size. Prevents loss spikes.
        @param: num_workers Number of CPU processes used for preprocessing.
        """
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = SimpleDataset(txt, tokenizer, max_len, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return dataloader

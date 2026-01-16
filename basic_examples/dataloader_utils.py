from .dataloader import SimpleDataset
import tiktoken     # Byte Pair Encoding
import torch

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
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return dataloader

def create_embeddings(embedding_size, inputs, vocab_size=50267):
        """
        Creates embeddings tensors for each input.
        
        @param embedding_size: Number of embeddings for each token.
        @param inputs: Tensor of inputs.
        @param vocab_size: Number of tokens in the vocabulary.
        """
        # Creates initial random values for each word in vocab
        token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
        # Gets corresponding values for each word in input
        token_embeddings = token_embedding_layer(inputs)
        
        # Create absolute positional embeddings
        context_len = len(inputs[0])
        pos_embedding_layer = torch.nn.Embedding(context_len, embedding_size)
        pos_embeddings = pos_embedding_layer(torch.arange(context_len))

        input_embeddings = token_embeddings + pos_embeddings
        return input_embeddings

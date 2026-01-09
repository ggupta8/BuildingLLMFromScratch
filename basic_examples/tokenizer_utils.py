from typing import List

import os
import re   

def read_txt_into_str(file_name: str) -> str:
    """Retrieves text from a given file."""
    file_path = os.getcwd() + file_name
    file_path = file_path.replace("/basic_examples", "")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_txt = f.read()
    return raw_txt

def split_txt_into_words(unsplit_txt: str) -> str:
    """Splits a text into an array of words.

    White space is not included. Punctionation are separate array elements.
    """
    split_txt_with_spaces = re.split(r'([,.:;?_!"()\']|--|\s)', unsplit_txt)
    final_split_txt = [item for item in split_txt_with_spaces if item.strip()]
    return final_split_txt

def create_vocabulary(txt_arr: List[str]):
    """Maps each word / symbol in text to a token ID."""
    # Alphabetize list of words and assign a token ID accordingly.
    sorted_word_list = sorted(set(txt_arr))
    # "endoftext" signifies a separation between text sources for the LLM to better
    # context. "unk" is for words that are not found in our vocabulary.
    sorted_word_list.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {word: idx for idx, word in enumerate(sorted_word_list)}
    return vocab
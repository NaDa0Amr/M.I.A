import json
from collections import Counter
from typing import List

class Vocabulary:
    PAD_TOKEN = '<pad>'  # index 0
    START_TOKEN = '<start>'  # index 1
    END_TOKEN = '<end>'  # index 2
    UNK_TOKEN = '<unk>'  # index 3
    
    def __init__(self, freq_threshold: int = 3):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            self.PAD_TOKEN: self.pad_idx,
            self.START_TOKEN: self.start_idx,
            self.END_TOKEN: self.end_idx,
            self.UNK_TOKEN: self.unk_idx
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.idx = 4
        
    def __len__(self) -> int:
        return len(self.word2idx)
        
    @property
    def pad_idx(self) -> int: return 0
    @property  
    def start_idx(self) -> int: return 1
    @property
    def end_idx(self) -> int: return 2
    @property
    def unk_idx(self) -> int: return 3
    
    @staticmethod
    def clean_caption(caption: str) -> list[str]:
        # Convert to lowercase
        caption = caption.lower()
        # Remove all punctuation
        import string
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        # Split on whitespace
        tokens = caption.split()
        # Keep only tokens with len > 1, EXCEPT keep 'a' and 'i'
        # Remove tokens that are purely numeric
        cleaned_tokens = [
            token for token in tokens
            if (len(token) > 1 or token in ('a', 'i')) and not token.isnumeric()
        ]
        return cleaned_tokens

    def build_vocabulary(self, captions: list[str]) -> None:
        frequencies = Counter()
        for caption in captions:
            tokens = self.clean_caption(caption)
            frequencies.update(tokens)
            
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
                
    def numericalize(self, caption: str) -> list[int]:
        tokens = self.clean_caption(caption)
        result = [self.start_idx]
        for token in tokens:
            result.append(self.word2idx.get(token, self.unk_idx))
        result.append(self.end_idx)
        return result
        
    def denumericalize(self, token_ids: list[int]) -> str:
        words = []
        for token_id in token_ids:
            if token_id == self.end_idx:
                break
            if token_id not in (self.pad_idx, self.start_idx):
                words.append(self.idx2word.get(token_id, self.UNK_TOKEN))
        return " ".join(words)
        
    def save(self, path: str) -> None:
        data = {
            "freq_threshold": self.freq_threshold,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.word2idx = data["word2idx"]
        # JSON keys are strings, convert to int for idx2word
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.idx = len(vocab.word2idx)
        return vocab

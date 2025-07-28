import string
import torch
import json

class koRKutTokenizer:
    def __init__(self, vocab_file, emptiness: int):
        try:
            with open (vocab_file, "r") as f:
                self.vocab = json.load(f)
                self.reverse_vocab = {k: v for v, k in self.vocab.items()}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid Dictionary {e}")
        
        self.emptiness = emptiness
        
    def preproccessingText(self, text):
        text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        return text

    def encode(self, text):
        text = self.preproccessingText(text=text)
        tokens = []
        
        for word in text.split():
            i = 0
            while i < len(word):
                found_match = False
                for j in range(len(word), i, -1):
                    sub_word = word[i:j]
                    if sub_word in self.vocab:
                        tokens.append(self.vocab[sub_word])
                        i = j
                        found_match = True
                        break
                if not found_match:
                    tokens.append(self.vocab["<unk>"])
                    i += 1
            tokens.append(self.emptiness)

        if not text.endswith(" "):
            tokens.pop()
        return torch.tensor(tokens)
    
    def tokenize(self, text):
        token_ids = self.encode(text)

        token_ids = token_ids.detach().numpy().tolist()

        return [self.reverse_vocab[id] for id in token_ids]
    
    def decode(self, ids):
        text = ""

        for id in ids:
            text += self.reverse_vocab[id]

        return text


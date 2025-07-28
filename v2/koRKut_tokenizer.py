import string
import torch
import json

def load_vocab_from_complex_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = {}

    if isinstance(data, dict):
        if all(isinstance(v, int) for v in data.values()):
            vocab = data

        elif "added_tokens" in data:
            for token_info in data["added_tokens"]:
                token_id = token_info.get("id")
                token_str = token_info.get("content")
                if token_id is not None and token_str is not None:
                    vocab[token_str] = token_id

            # Eğer model içinde ayrıca vocab varsa onu da ekleyelim
            if "model" in data and "vocab" in data["model"]:
                vocab.update(data["model"]["vocab"])
    else:
        raise ValueError("Unsupported vocab file format")

    if not vocab:
        raise ValueError("No vocab entries found in the provided file")

    return vocab


class koRKutTokenizer:
    def __init__(self, vocab_file, emptiness: int):
        try:
            self.vocab = load_vocab_from_complex_json(vocab_file)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        except Exception as e:
            raise ValueError(f"Invalid Dictionary or file format: {e}")
        
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
                    tokens.append(self.vocab.get("<unk>", 0))
                    i += 1
            tokens.append(self.emptiness)

        if not text.endswith(" "):
            tokens.pop()
        return torch.tensor(tokens)
    
    def tokenize(self, text):
        token_ids = self.encode(text)
        token_ids = token_ids.detach().numpy().tolist()
        return [self.reverse_vocab.get(id, "<unk>") for id in token_ids]
    
    def decode(self, ids):
        text = ""
        for id in ids:
            text += self.reverse_vocab.get(id, "<unk>")
        return text

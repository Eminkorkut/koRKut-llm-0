import torch
from torch.utils.data import DataLoader, Dataset

pad_id = 63

class TextDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, context_length: int, stride: int):
        super().__init__()

        self.inputs = []
        self.targets = []

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i : i + context_length]
            target_chunk = token_ids[i + 1 : i + context_length + 1]

            input_chunk = input_chunk[:context_length]
            target_chunk = target_chunk[:context_length]

            pad_len = context_length - input_chunk.size(0)

            if pad_len > 0:
                input_chunk = torch.cat([input_chunk, torch.full((pad_len,), pad_id, dtype=torch.long)])
                target_chunk = torch.cat([target_chunk, torch.full((pad_len,), pad_id, dtype=torch.long)])

            self.inputs.append(input_chunk)
            self.targets.append(target_chunk)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]



def create_data_loader(token_ids: list, context_length: int, stride: int,
                       batch_size: int, shuffle: bool = True, device: str = "cpu"):
  dataset = TextDataset(token_ids, context_length, stride)
  dataloader = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=shuffle,
      generator=torch.Generator(device=device)
    )
  
  return dataloader


  
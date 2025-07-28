import torch
import torch.nn as nn

class koRKutTrain:
    def __init__(self, model, dataloader, optimizer, loss_fn, device: str="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train(self, epochs: int=10, log_interval=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % log_interval == 0:
                avg_loss = total_loss / len(self.dataloader)
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
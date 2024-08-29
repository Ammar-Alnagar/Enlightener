from state_spaces.models.mamba import Mamba
import torch

class MambaModel:
    def __init__(self, config):
        self.model = Mamba.from_pretrained(config['model_name'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def encode(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.model.tokenizer(text, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

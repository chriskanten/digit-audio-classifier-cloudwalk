import torch
import torch.nn as nn

class AudioDigitClassifier(nn.Module):
    def __init__(self, input_dim=13, num_classes=10, hidden_dim=64):
        super(AudioDigitClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in [hidden_dim, hidden_dim // 2]:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.5)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
       
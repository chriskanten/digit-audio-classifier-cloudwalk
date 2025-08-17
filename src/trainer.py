import torch
import numpy as np
import torch.nn as nn                                                   
import torch.optim as optim
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader, epochs=10):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for train_features,train_labels in tqdm(train_loader, desc="Training"):
            train_features, train_labels = train_features.to(self.device), train_labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(train_features)
            loss = self.criterion(outputs, train_features)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += train_labels.size(0)
            correct += (predicted == train_labels).sum().item()
            
        return total_loss / len(train_loader), 100 * correct / total

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        
        with torch.no_grad():
            for test_features, test_labels in tqdm(test_loader, desc="Evaluating"):
                test_features, test_labels = test_features.to(self.device), test_labels.to(self.device)

                outputs = self.model(test_features)
                loss = self.criterion(outputs, test_features)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        return total_loss / len(test_loader), 100 * correct / total
        

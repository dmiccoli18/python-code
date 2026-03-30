import torch
from torch import nn
import os

class NeuralNetwork(nn.Module):
    def __init__(self,num_weights):
        super().__init__()
        self.num_weights = num_weights
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, num_weights),
            nn.ReLU(),
            nn.Linear(num_weights,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
    
    # model save
    def model_save(model):
        currpath = os.getcwd()
        model_path = currpath + "\saved_models"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        os.chdir(model_path)
        torch.save(model.state_dict, f"model.pth") # add model weights

    # model load
    def model_load(model):
        currpath = os.getcwd()
        model_path = currpath + "\saved_models"
        if not os.path.exists(model_path):
            return None
        else:
            os.chdir(model_path)
            model.load_state_dict(torch.load("model.pth")) # add model weights
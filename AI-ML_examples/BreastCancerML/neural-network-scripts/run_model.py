
import data_preprocessing as dp
import model as net
import train_model as tm
import model_evaluation as eval
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.DoubleTensor)

train, valid, test = dp.data_split()

train, X_train, Y_train = dp.scale_dataset(train, oversample=True)
valid, X_valid, Y_valid = dp.scale_dataset(valid, oversample=False)
test, X_test, Y_test = dp.scale_dataset(test, oversample=False)

model = net.NeuralNetwork(9)
model.to(device)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model = tm.train_model(model, X_train, Y_train, loss_fn, optimizer, epochs=20)

print(f"Validation Accuracy: {eval.model_eval(model,X_valid,Y_valid)}")
print("-------------------------------")
print(f"Test Accuracy: {eval.model_eval(model,X_test,Y_test)}")
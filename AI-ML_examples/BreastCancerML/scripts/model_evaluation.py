import torch

def model_eval(model,X_data,Y_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        correct = 0
        for idx,x in enumerate(X_data):
            X, Y = torch.tensor(x).to(device), torch.tensor([Y_data[idx]]).to(device).to(torch.float64)
    
            # Forward pass: Compute prediction and loss
            pred = model(X)
            # accuracy
            if pred[0] > .5:
                pred[0] = 1
            else:
                pred[0] = 0
            correct += (pred == Y.int()).sum().item()
    
    return correct/len(X_data)
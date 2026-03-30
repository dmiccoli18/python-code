import torch

def train_model(model, X_train, Y_train, loss_fn, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() # set model to training mode
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        correct = 0
        for idx,x in enumerate(X_train):
            X, Y = torch.tensor(x).to(device), torch.tensor([Y_train[idx]]).to(device).to(torch.float64)
    
            # Forward pass: Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, Y)
    
            # Backward pass: Compute gradients and update weights
            optimizer.zero_grad() # clear previous gradients
            loss.backward() # backpropagate loss
            optimizer.step() # update weights
    
            # accuracy
            if pred[0] > .5:
                pred[0] = 1
            else:
                pred[0] = 0
            correct += (pred == Y.int()).sum().item()
    
            # loss
        loss = loss.item()
        print(f"loss: {loss:>7f}")
        print(correct/len(X_train))
    print("Done!")

    return model
import torch

# Pytorch style train and test pipelines.
def train(loader, model, optimizer, criterion=None):
    model.train()
    criterion = model.loss_fn()
    base_loss = 0
    train_loss = 0
    for i, data in enumerate(loader): # Iterate in batches over the training dataset.
        X, y, meta = data # meta is the metadata
        tau = meta['tau'][0]
        # Perform a single forward pass.
        try:
          out, states = model(X, tau, states)
        except:
          out, states = model(X, tau) # states=None at start of epoch
        states = tuple(map(lambda x: x.detach(), states))
        # Compute the baseline loss
        base = criterion(X, y)/(1 + tau) # loss if model predicts y(t) for y(t+1)
        # Compute the training loss.
        loss = model.elbo_loss + criterion(out, y)/(1 + tau) 
        loss.backward()  # Derive gradients.
        # TODO: figure out if gradient clipping is necessary.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) 
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        # Store train and baseline loss.
        base_loss += base.detach().item()
        train_loss += loss.detach().item()
    # return mean train and baseline losses
    return train_loss/(i+1), base_loss/(i+1)

def test(loader, model, criterion=None):
    model.eval()
    criterion = model.loss_fn()
    base_loss = 0
    val_loss = 0
    for i, data in enumerate(loader): # Iterate in batches over the training/test dataset.
        X, y, meta = data # meta is the metadata
        tau = meta['tau'][0]
        # Perform a single forward pass.
        try:
          out, states = model(X, tau, states)
        except:
          out, states = model(X, tau) # states=None at start of epoch
        states = tuple(map(lambda x: x.detach(), states))
        # Compute the baseline loss.
        base = criterion(X, y)/(1 + tau) # loss if model predicts y(t) for y(t+1)
        # Store the validation and baseline loss.
        base_loss += base.detach().item()
        val_loss += model.elbo_loss.detach().item() + criterion(out, y).detach().item()/(1 + tau.item()) 
    # return mean validation and baseline losses
    return val_loss/(i+1), base_loss/(i+1) # Return the mean validation loss.
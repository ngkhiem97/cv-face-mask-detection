import torch.optim as optim

def set_learning_rate(optimizer, epoch, base_lr):
    # This function is inspired by assigment 2
    lr = base_lr*0.3**(epoch//3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_optimizer(model, optimizer_type, lr):
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer_type should be 'SGD' or 'Adam'")
    return optimizer
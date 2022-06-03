import os
import json
import torch

from torchvision import models
from utils.training import *

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    train_loader, validate_loader = load_data()
    net = models.alexnet(pretrained=True)
    net.classifier[6] = nn.Linear(4096, 2)
    learning_path = './learning.json'
    assert os.path.exists(learning_path), "file: '{}' dose not exist.".format(learning_path)
    with open(learning_path, "r") as f:
        learning = json.load(f)
    for learning_item in learning:
        for lr in learning[learning_item]["lr"]:
            train(net, device, train_loader, validate_loader, learning_item, lr, 20, "AlexNet_pretrained")

if __name__ == '__main__':
    main()

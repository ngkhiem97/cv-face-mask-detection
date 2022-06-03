import os
import json
import torch

from models.mobilenet import MobileNet
from utils.training import *

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    train_loader, validate_loader = load_data()
    net = MobileNet(num_classes=2, init_weights=True)
    learning_path = './learning.json'
    assert os.path.exists(learning_path), "file: '{}' dose not exist.".format(learning_path)
    with open(learning_path, "r") as f:
        learning = json.load(f)
    for learning_item in learning:
        for lr in learning[learning_item]["lr"]:
            train(net, device, train_loader, validate_loader, learning_item, lr, 10, "MobileNet")

if __name__ == '__main__':
    main()

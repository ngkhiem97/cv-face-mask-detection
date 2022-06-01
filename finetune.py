import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from models.alexnet import AlexNet
import time

from torchvision import models

def train(model, device, train_loader, validate_loader, optimizer_type, lr, epochs):
    print(f"Training start with {optimizer_type} and lr={lr}")
    model.to(device)
    loss_function = nn.CrossEntropyLoss()

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("optimizer_type should be 'SGD' or 'Adam'")

    since = time.time()
    save_path = './models/AlexNet_pretrained_'+optimizer_type+'_'+str(lr).replace(".", "-")+'.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    training_loss = []
    val_accuracy = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # print statistics
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / validate_loader.dataset.__len__()
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        training_loss.append(running_loss / train_steps)
        val_accuracy.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print(f'Finished training for {optimizer_type} and lr={lr}')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    
    model_name = 'AlexNet_pretrained_'+optimizer_type+'_'+str(lr).replace(".", "-")
    model_dict ={model_name:{'training time': time_elapsed, 'best accuracy': best_acc, 'training loss': training_loss, 'validation accuracy': val_accuracy}}
    json_str = json.dumps(model_dict, indent=4)
    with open('./log/training_'+model_name+'.json', 'w') as json_file:
        json_file.write(json_str)
    
    # with open('./log/training_AlexNet_'+optimizer_type+'_'+str(lr).replace(".", "-")+'.txt', 'w') as f:
    #     f.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    #     f.write(f'Best val_accuracy: {best_acc}\n')
    #     f.write(f'Training loss: {training_loss}\n')
    #     f.write(f'Validation accuracy: {val_accuracy}\n')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.getcwd()
    image_path = os.path.join(data_root, "dataset")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {
    #     "0": "face",
    #     "1": "face_mask"
    # }
    categories_list = train_dataset.class_to_idx
    categories_dict = dict((val, key) for key, val in categories_list.items())
    print("categories_dict: {}".format(categories_dict))

    # write dict into json file
    json_str = json.dumps(categories_dict, indent=4)
    with open('categories.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    net.fc = nn.Linear(num_ftrs, 2)

    learning_path = './learning.json'
    assert os.path.exists(learning_path), "file: '{}' dose not exist.".format(learning_path)
    with open(learning_path, "r") as f:
        learning = json.load(f)
    for learning_item in learning:
        for lr in learning[learning_item]["lr"]:
            train(net, device, train_loader, validate_loader, learning_item, lr, 10)

    # net.to(device)
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # # start training
    # epochs = 10
    # save_path = './models/AlexNet.pth'
    # best_acc = 0.0
    # train_steps = len(train_loader)
    # for epoch in range(epochs):
    #     net.train()
    #     running_loss = 0.0
    #     train_bar = tqdm(train_loader, file=sys.stdout)
    #     for step, data in enumerate(train_bar):
    #         images, labels = data
    #         optimizer.zero_grad()
    #         outputs = net(images.to(device))
    #         loss = loss_function(outputs, labels.to(device))
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()

    #         # print statistics
    #         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
    #                                                                  epochs,
    #                                                                  loss)

    #     # validate
    #     net.eval()
    #     acc = 0.0  # accumulate accurate number / epoch
    #     with torch.no_grad():
    #         val_bar = tqdm(validate_loader, file=sys.stdout)
    #         for val_data in val_bar:
    #             val_images, val_labels = val_data
    #             outputs = net(val_images.to(device))
    #             predict_y = torch.max(outputs, dim=1)[1]
    #             acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    #     val_accurate = acc / val_num
    #     print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
    #           (epoch + 1, running_loss / train_steps, val_accurate))

    #     if val_accurate > best_acc:
    #         best_acc = val_accurate
    #         torch.save(net.state_dict(), save_path)

    # print('Finished Training')

if __name__ == '__main__':
    main()

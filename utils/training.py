import torch.optim as optim
from torchvision import transforms, datasets
import os
import torch
import json
import torch.nn as nn
import time
from tqdm import tqdm
import sys

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

def load_data():
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
    # {
    #     "0": "face",
    #     "1": "face_mask"
    # }
    categories_list = train_dataset.class_to_idx
    categories_dict = dict((val, key) for key, val in categories_list.items())
    # write dict into json file
    json_str = json.dumps(categories_dict, indent=4)
    with open('categories.json', 'w') as json_file:
        json_file.write(json_str)
    print("categories_dict: {}".format(categories_dict))

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(len(train_dataset),
                                                                           len(validate_dataset)))
    return train_loader, validate_loader


def train(model, device, train_loader, validate_loader, optimizer_type, lr, epochs, name="DeepLearning"):
    print(f"Training start with {optimizer_type} and lr={lr}")
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_type, lr)
    since = time.time()
    save_path = './models/'+name+'_'+optimizer_type+'_'+str(lr).replace(".", "-")+'.pth'
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

    
    model_name = name+'_'+optimizer_type+'_'+str(lr).replace(".", "-")
    model_dict ={model_name:{'training time': time_elapsed, 'best accuracy': best_acc, 'training loss': training_loss, 'validation accuracy': val_accuracy}}
    json_str = json.dumps(model_dict, indent=4)
    with open('./log/training_'+model_name+'.json', 'w') as json_file:
        json_file.write(json_str)
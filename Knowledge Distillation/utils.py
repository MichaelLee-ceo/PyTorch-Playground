import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, data_loader, optimizer, device):
    train_total, train_correct = 0, 0
    model.train()
    for idx, (x, label) in enumerate(data_loader):
        x, label = x.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(x)

        predicted = torch.argmax(output.data, 1)
        train_total += label.size(0)
        train_correct += (predicted == label).sum().item()

        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
    return train_correct / train_total

def evaluate(model, data_loader, device):
    val_total, val_correct = 0, 0
    with torch.no_grad():
        model.eval()
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            predicted = torch.argmax(outputs.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    return val_correct / val_total

def getDataset(dataset, label):
    indices = torch.nonzero(torch.Tensor(dataset.targets) == label).squeeze()
    trainset = torch.utils.data.Subset(dataset, indices)

    train_size = int(len(trainset) * 0.95)
    val_size = len(trainset) - train_size

    train_set, val_set = torch.utils.data.random_split(trainset, [train_size, val_size])
    return train_set, val_set

def getDevice():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}, {torch.cuda.get_device_name(device)}')
    print(torch.cuda.get_device_properties(device), '\n')
    return device

def mixup(img, label, alpha=0.2):
    lamb = np.random.beta(alpha, alpha)
    mixup_idx = torch.randperm(len(img))

    mixup_img = img[mixup_idx]
    mixup_label = label[mixup_idx]

    labels = F.one_hot(label).long()
    mixup_labels = F.one_hot(mixup_label).long()
    return lamb * img + (1 - lamb) * mixup_img, lamb * labels + (1 - lamb) * mixup_labels

def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

def loss_fn_kd(outputs, labels, teacher_outputs, T=4, alpha=0.9):
    KD_loss = nn.KLDivLoss(reduction="batchmean")(nn.functional.log_softmax(outputs/T, dim=1), 
                             nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha*T*T)
    KD_loss += nn.functional.cross_entropy(outputs, labels) * (1.0 - alpha)
    return KD_loss

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_train_result(num_epochs, train_loss, train_acc, val_loss, val_acc, name):
    x_range = np.arange(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(x_range, train_loss, label='train loss')
    plt.plot(x_range, val_loss, label='validation loss')

    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(x_range, train_acc, label='train acc')
    plt.plot(x_range, val_acc, label='validation acc')
    plt.legend()
    plt.savefig('./figures/' + name + '.png')
    # plt.show()
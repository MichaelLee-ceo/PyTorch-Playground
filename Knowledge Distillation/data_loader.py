import torch
import torchvision
from myDataset import MyDataset

torch.manual_seed(0)

def DataLoader(batch_size=128, train_val_split=0.8, mixup=True):
    data_transform = {
        'basic': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5), (0.5)),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        'augment':  torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
        ]),
    }

    dataset = torchvision.datasets.CIFAR10(root='./data', download=False, train=True, transform=data_transform['basic'])
    testset = torchvision.datasets.CIFAR10(root='./data', download=False, train=False, transform=data_transform['basic'])

    train_size = int(len(dataset) * train_val_split)
    val_size= len(dataset) - train_size
    trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

    new_trainset = MyDataset(trainset, transform=data_transform['augment'], mixup=False)
    print('Original trainset: {}'.format(len(trainset)))

    if mixup:
        mixup_dataset = MyDataset(trainset, transform=data_transform['augment'], mixup=True)
        new_trainset = torch.utils.data.ConcatDataset([new_trainset, mixup_dataset])
        print('[+] Create Mixup data augmentation: {}'.format(len(mixup_dataset)))

    print('Total trainset: {}'.format(len(new_trainset)))

    train_loader = torch.utils.data.DataLoader(new_trainset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
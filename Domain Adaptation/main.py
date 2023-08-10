import argparse
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from mnist_m import MNISTM
from torchvision import datasets
from torchvision.utils import make_grid, save_image
from model import MyNet
from utils import *

sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.003, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--domain_adaptation', action="store_true")
parser.add_argument('--scheduler', default="constant", type=str)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = args.num_epochs
batch_size = args.batch_size
half_batch = int(batch_size / 2)
lr = args.lr

transform_mnist = transforms.Compose([
    GrayscaleToRgb(),
    transforms.ToTensor(),
])

transform_mnistm = transforms.Compose([
    transforms.ToTensor(),
])

''' Construct Source, Target Dataset by MNIST & MNIST-M '''
trainset_0 = datasets.MNIST(root='./data', download=False, train=True, transform=transform_mnist)
testset_0 = datasets.MNIST(root='./data', download=False, train=False, transform=transform_mnist)

trainset_1 = MNISTM(root='./data', download=False, train=True, transform=transform_mnistm)
testset_1 = MNISTM(root='./data', download=False, train=False, transform=transform_mnistm)

# trainset_1 = datasets.FashionMNIST(root='./data', download=True, train=True, transform=transform)
# testset_1 = datasets.FashionMNIST(root='./data', download=True, train=False, transform=transform)

''' Construct Source, Target DataLoader by MNIST & MNIST-M '''
trainLoader_0 = DataLoader(trainset_0, batch_size=half_batch, shuffle=True, drop_last=True)
testLoader_0 = DataLoader(testset_0, batch_size=batch_size, shuffle=False)

trainLoader_1 = DataLoader(trainset_1, batch_size=half_batch, shuffle=True, drop_last=True)
testLoader_1 = DataLoader(testset_1, batch_size=batch_size, shuffle=False)

# save_image(next(iter(trainLoader_0))[0][:32], 
#            './figures/mnist.png', nrow=8, normalize=True)
# save_image(next(iter(trainLoader_1))[0][:32], 
#            './figures/mnist_m.png', nrow=8, normalize=True)

model = MyNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
domainScheduler = DomainScheduler(max_iter=num_epochs, alpha=10, constant=args.scheduler)

state = {
    "acc": 0,
    "model": model.state_dict(),
}

for epoch in range(num_epochs):
    domain_loss = 0

    ''' Train and Test on Source Dataset '''
    if args.domain_adaptation:
        train_loss, domain_loss, train_acc = train_double(model, trainLoader_0, trainLoader_1, optimizer, lrScheduler, domainScheduler, epoch, device, args)
    else:
        train_loss, train_acc = train_single(model, trainLoader_0, optimizer, device, args)

    test_acc = evaluate(model, testLoader_0, device, args)
    tqdm.write('Epoch: [{}/{}] Loss: {:.4f}, Domain_loss: {:.4f}, Train_Acc: {:.2f}, Test_Acc: {:.2f}'
          .format(epoch+1, num_epochs, train_loss, domain_loss, train_acc, test_acc))
    
    ''' Test on Target Dataset '''
    target_acc = evaluate(model, testLoader_1, device, args)
    if target_acc > state["acc"]:
        state["acc"] = target_acc
        state["model"] = model.state_dict()
    tqdm.write("[Target Dataset] Acc: {:.2f}".format(target_acc))

# save model
if args.domain_adaptation:
    torch.save(state["model"], "model_da.pkt")
else:
    torch.save(state["model"], "model_source.pkt")
print("Best Acc: {:.2f}".format(state["acc"]))
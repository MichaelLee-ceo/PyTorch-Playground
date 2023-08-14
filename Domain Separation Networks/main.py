import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from mnist_m import MNISTM
from torchvision import datasets
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', default=50, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--private_dim', default=32, type=int)
parser.add_argument('--shared_dim', default=64, type=int)
parser.add_argument('--ckpt_path', default="./checkpoint/", type=str)
parser.add_argument('--alpha', default=0.01, type=float)
parser.add_argument('--beta', default=0.05, type=float)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.ckpt_path, exist_ok=True)

num_epochs = args.num_epochs
batch_size = args.batch_size
half_batch = int(batch_size / 2)
lr = args.lr
private_dim = args.private_dim
shared_dim = args.shared_dim
    
''' Data Preprocessing '''
transform_mnist = transforms.Compose([
    GrayscaleToRgb(),
    transforms.ToTensor(),
])
transform_mnistm = transforms.Compose([
    transforms.ToTensor(),
])

''' Construct Source (MNIST) -> Target (MNIST-M) Dataset '''
trainset_0 = datasets.MNIST(root='./data', download=False, train=True, transform=transform_mnist)
testset_0 = datasets.MNIST(root='./data', download=False, train=False, transform=transform_mnist)

trainset_1 = MNISTM(root='./data', download=False, train=True, transform=transform_mnistm)
testset_1 = MNISTM(root='./data', download=False, train=False, transform=transform_mnistm)

''' Construct Source (MNIST) -> Target (MNIST-M) DataLoader '''
trainLoader_0 = DataLoader(trainset_0, batch_size=half_batch, shuffle=True, drop_last=True)
testLoader_0 = DataLoader(testset_0, batch_size=batch_size, shuffle=False)

trainLoader_1 = DataLoader(trainset_1, batch_size=half_batch, shuffle=True, drop_last=True)
testLoader_1 = DataLoader(testset_1, batch_size=batch_size, shuffle=False)

''' Construct Domain Separation Network'''
model = {
    "Source_encoder": Private_Encoder(out_dim=private_dim).to(device),
    "Target_encoder": Private_Encoder(out_dim=private_dim).to(device),
    "Shared_encoder": Shared_Encoder(out_dim=shared_dim).to(device),
    "Shared_decoder": Shared_Decoder().to(device),
    "Classifier": Classifier().to(device),
    "Domain_classifier": Domain_Classifier().to(device),
}

optimizer = torch.optim.SGD(
    list(model["Source_encoder"].parameters()) + \
    list(model["Target_encoder"].parameters()) + \
    list(model["Shared_encoder"].parameters()) + \
    list(model["Shared_decoder"].parameters()) + \
    list(model["Classifier"].parameters()) + \
    list(model["Domain_classifier"].parameters())
, lr=lr, momentum=0.9, weight_decay=5e-4)
domain_scheduler = DomainScheduler(num_epochs, 10)

state = {
    "acc": 0,
    "model": model,
}

for epoch in range(num_epochs):
    result = train(model, trainLoader_0, trainLoader_1, optimizer, domain_scheduler, epoch, device, args)
    print("Epoch: {}/{}, loss: {:.4f}, acc: {:.4f}, sdiff_loss: {:.4f}, tdiff_loss: {:.4f}, sre_loss: {:.4f}, tre_loss: {:.4f}, similarity_loss: {:.4f}, label_loss: {:.4f}".format(
                epoch+1,
                num_epochs,
                result["loss"],
                result["acc"],
                result["sdiff_loss"],
                result["tdiff_loss"],
                result["sre_loss"],
                result["tre_loss"],
                result["similarity_loss"],
                result["label_loss"],
          ))
    
    test_acc = evaluate(model, testLoader_0, device, args)
    target_test_acc = evaluate(model, testLoader_1, device, args)

    if target_test_acc > state["acc"]:
        state["acc"] = target_test_acc
        state["model"] = model
    print("[Source] Test acc: {:.4f}, [Target] Test acc: {:.4f}".format(test_acc, target_test_acc))

torch.save(state, os.path.join(args.ckpt_path, "best_model.pth"))
print("Best acc: {:.4f}".format(state["acc"]))
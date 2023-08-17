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
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--ckpt_path', default="./checkpoint/", type=str)
parser.add_argument('--figure_path', default="./figures/", type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.figure_path, exist_ok=True)

''' Data Preprocessing '''
transform_mnist = transforms.Compose([
    GrayscaleToRgb(),
    transforms.ToTensor(),
])
transform_mnistm = transforms.Compose([
    transforms.ToTensor(),
])

''' Construct Source (MNIST) -> Target (MNIST-M) Dataset '''
testset_0 = datasets.MNIST(root='./data', download=False, train=False, transform=transform_mnist)
testset_1 = MNISTM(root='./data', download=False, train=False, transform=transform_mnistm)
testLoader_0 = DataLoader(testset_0, batch_size=args.batch_size, shuffle=True)
testLoader_1 = DataLoader(testset_1, batch_size=args.batch_size, shuffle=True)

ckpt = torch.load(args.ckpt_path + "best_model.pth")
model = {
    "Source_encoder": ckpt["model"]["Source_encoder"].to(device),
    "Target_encoder": ckpt["model"]["Target_encoder"].to(device),
    "Shared_encoder": ckpt["model"]["Shared_encoder"].to(device),
    "Shared_decoder": ckpt["model"]["Shared_decoder"].to(device)
}
best_acc = ckpt["acc"]

visualize(model, testLoader_0, testLoader_1, device, best_acc, args)
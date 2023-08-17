import os
import argparse
import torch
import random
import numpy as np
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
parser.add_argument('--train_on', default="source", type=str)
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
testLoader_0 = DataLoader(testset_0, batch_size=args.batch_size, shuffle=False, generator=torch.manual_seed(args.seed))
testLoader_1 = DataLoader(testset_1, batch_size=args.batch_size, shuffle=False, generator=torch.manual_seed(args.seed))

if args.train_on == "source":
    ckpt = torch.load(args.ckpt_path + "model_source.pth")
elif args.train_on == "da":
    ckpt = torch.load(args.ckpt_path + "model_da.pth")
else:
    raise Exception("Non-supported source type.")

visualize(ckpt, testLoader_0, testLoader_1, device, args)
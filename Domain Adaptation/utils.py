import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class DomainScheduler:
    def __init__(self, max_iter, alpha=0.75, constant=True):
        self.max_iter = max_iter
        self.alpha = alpha
        self.constant = constant

    def __call__(self, num_iter):
        if self.constant:
            return 1
        else:
            p = num_iter / self.max_iter
            return 2. / (1. + np.exp(-self.alpha * p)) - 1.


def train_single(model, dataLoader_0, optimizer, device, args):
    total_loss, total_correct = 0, 0
    model.train()

    for idx, (x0, label0) in enumerate(tqdm(dataLoader_0, leave=False)):
        x0, label0 = x0.to(device), label0.to(device)
        output0, _ = model(x0)

        predict = torch.argmax(output0.data, 1)
        total_correct += (predict == label0).sum().item()

        loss = F.cross_entropy(output0, label0)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataLoader_0), total_correct / len(dataLoader_0.dataset)


def train_double(model, dataLoader_0, dataLoader_1, optimizer, lr_scheduler, da_scheduler, epoch, device, args):
    total_loss, total_correct = 0, 0
    total_domain_loss = 0
    model.train()

    weight = da_scheduler(epoch)
    print("epoch: {}, weight: {}".format(epoch+1, weight))

    num_batches = min(len(dataLoader_0), len(dataLoader_1))
    for idx, ((x0, label0), (x1, _)) in enumerate(tqdm(zip(dataLoader_0, dataLoader_1), leave=False, total=num_batches)):
        x0, label0 = x0.to(device), label0.to(device)
        x1 = x1.to(device)

        output, domain_source = model(x0)
        _, domain_target = model(x1)

        domain_x = torch.cat([domain_source, domain_target])
        domain_y = torch.cat([
            torch.ones_like(domain_source),
            torch.zeros_like(domain_target)
        ])

        predict = torch.argmax(output.data, 1)
        total_correct += (predict == label0).sum().item()

        ce_loss = F.cross_entropy(output, label0)
        feature_loss = F.binary_cross_entropy_with_logits(domain_x, domain_y)
        loss = ce_loss + feature_loss * weight
        
        total_loss += ce_loss.item()
        total_domain_loss += feature_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lr_scheduler.step(total_loss / len(dataLoader_0))
    return total_loss / len(dataLoader_0), total_domain_loss / len(dataLoader_0), total_correct / len(dataLoader_0.dataset)


def evaluate(model, dataLoader, device, args):
    total_correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in dataLoader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)

            predict = torch.argmax(output.data, 1)
            total_correct += (predict == target).sum().item()
    return total_correct / len(dataLoader.dataset)
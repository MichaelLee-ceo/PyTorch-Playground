import wandb
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from sklearn.manifold import TSNE


class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)


class DomainScheduler:
    def __init__(self, max_iter, alpha=0.75, scheduler="constant"):
        self.max_iter = max_iter
        self.alpha = alpha
        self.scheduler = scheduler

    def __call__(self, num_iter):
        if self.scheduler == "constant":
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
    wandb.log({"domain_weight": weight})
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
        domain_loss = F.binary_cross_entropy_with_logits(domain_x, domain_y)
        loss = ce_loss + domain_loss * weight
        
        total_loss += ce_loss.item()
        total_domain_loss += domain_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # lr_scheduler.step(total_loss / len(dataLoader_0))
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


def visualize(ckpt, dataLoader1, dataLoader2, device, args):
    ''' Visualize the feature embeddings of the feature extractor '''
    source = next(iter(dataLoader1))
    target = next(iter(dataLoader2))
    model, best_acc = ckpt["model"].to(device), ckpt["acc"]
    model.eval()

    with torch.no_grad():
        source_x, source_y = source[0].to(device), source[1]
        target_x, target_y = target[0].to(device), target[1]

        source_embedding, _ = model(source_x)
        target_embedding, _ = model(target_x)

        embeddings = TSNE(n_components=2).fit_transform(torch.cat([source_embedding, target_embedding]).cpu().numpy())
        y = torch.cat([source_y, target_y]).cpu().numpy()

        x_min, x_max = embeddings.min(0), embeddings.max(0)
        X_norm = (embeddings - x_min) / (x_max - x_min)
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            if i < (args.batch_size/2):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i].item()), color="dodgerblue", fontdict={'weight': 'bold', 'size': 10})
            else:
                plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i].item()), color="silver", fontdict={'weight': 'bold', 'size': 10})
        
        figname = ""
        if args.train_on == "source":
            plt.title("t-SNE on MNIST & MNIST-M, {:.2f}%".format(best_acc*100))
            figname = args.figure_path + "tsne_no_da.png"
        elif args.train_on == "da":
            plt.title("(DANN) t-SNE on MNIST & MNIST-M, {:.2f}%".format(best_acc*100))
            figname = args.figure_path + "tsne_da.png"

        plt.savefig(figname)
        print("Source image saved to {}".format(figname))
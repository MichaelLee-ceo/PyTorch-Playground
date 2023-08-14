import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

''' Convert a grayscale image to rgb '''
class GrayscaleToRgb:
    def __call__(self, image):
        image = np.array(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)

''' Scheduler for weight of similarity loss '''
class DomainScheduler:
    def __init__(self, max_iter, alpha=0.75, scheduler="da"):
        self.max_iter = max_iter
        self.alpha = alpha
        self.scheduler = scheduler

    def __call__(self, num_iter):
        if self.scheduler == "da":
            p = num_iter / self.max_iter
            return 2. / (1. + np.exp(-self.alpha * p)) - 1.
        else:
            return 1

''' Scale-Invariant Mean Squared Error '''
class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.mean(diffs.pow(2)) - (torch.sum(diffs).pow(2) / (n ** 2))
        return simse


class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss


def train(model, dataLoader_0, dataLoader_1, optimizer, domain_scheduler, epoch, device, args):
    record = {
        "loss": 0,
        "acc": 0,
        "sdiff_loss": 0,
        "tdiff_loss": 0,
        "sre_loss": 0,
        "tre_loss": 0,
        "similarity_loss": 0,
        "label_loss": 0,
    }
    total = 0

    source_encoder = model["Source_encoder"]
    target_encoder = model["Target_encoder"]
    shared_encoder = model["Shared_encoder"]
    shared_decoder = model["Shared_decoder"]
    class_classifier = model["Classifier"]
    domain_classifier = model["Domain_classifier"]

    source_encoder.train()
    target_encoder.train()
    shared_encoder.train()
    shared_decoder.train()
    class_classifier.train()
    domain_classifier.train()

    weight = domain_scheduler(epoch)
    print("epoch: {}, weight: {}".format(epoch+1, weight))

    num_batches = min(len(dataLoader_0), len(dataLoader_1))
    for idx, ((x0, label0), (x1, _)) in enumerate(tqdm(zip(dataLoader_0, dataLoader_1), total=num_batches, leave=False)):
        x0, label0 = x0.to(device), label0.to(device)
        x1 = x1.to(device)

        source_private = source_encoder(x0)          # output shape: [batch, 100]
        source_shared = shared_encoder(x0)          # output shape: [batch, 100]

        target_private = target_encoder(x1)          # output shape: [batch, 100]
        target_shared = shared_encoder(x1)          # output shape: [batch, 100]

        ''' Compute Difference Loss (private_features & shared_features) '''
        source_diff_loss = DiffLoss()(source_private, source_shared)
        target_diff_loss = DiffLoss()(target_private, target_shared)
        record["sdiff_loss"] += source_diff_loss.item()
        record["tdiff_loss"] += target_diff_loss.item()

        ''' Compute Reconstruction Loss '''
        source_re = shared_decoder(source_private + source_shared)
        target_re = shared_decoder(target_private + target_shared)
        source_re_loss = SIMSE()(source_re, x0)
        target_re_loss = SIMSE()(target_re, x1)
        record["sre_loss"] += source_re_loss.item()
        record["tre_loss"] += target_re_loss.item()

        ''' Compute Similarity Loss '''
        domain_source = domain_classifier(source_shared)
        domain_target = domain_classifier(target_shared)
        domain_x = torch.cat([domain_source, domain_target])
        domain_y = torch.cat([
            torch.ones_like(domain_source),
            torch.zeros_like(domain_target)
        ])
        similarity_loss = F.binary_cross_entropy_with_logits(domain_x, domain_y)
        record["similarity_loss"] += similarity_loss.item()

        ''' Compute Label Loss '''
        output = class_classifier(source_shared)
        predict = torch.argmax(output.data, 1)
        record["acc"] += (predict == label0).sum().item()
        total += label0.shape[0]

        label_loss = F.cross_entropy(output, label0)
        record["label_loss"] += label_loss.item()

        loss = (source_re_loss + target_re_loss) * args.alpha + \
               (source_diff_loss + target_diff_loss) * args.beta  + \
               (similarity_loss) * weight + label_loss
        record["loss"] += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    record["loss"] /= num_batches
    record["acc"] /= total
    record["sdiff_loss"] /= num_batches
    record["tdiff_loss"] /= num_batches
    record["sre_loss"] /= num_batches
    record["tre_loss"] /= num_batches
    record["similarity_loss"] /= num_batches
    record["label_loss"] /= num_batches
    return record


def evaluate(model, dataLoader, device, args):
    shared_encoder = model["Shared_encoder"].eval()
    classifier = model["Classifier"].eval()

    total_correct = 0
    with torch.no_grad():
        for data, target in dataLoader:
            data, target = data.to(device), target.to(device)
            output = classifier(shared_encoder(data))

            predict = torch.argmax(output.data, 1)
            total_correct += (predict == target).sum().item()
    return total_correct / len(dataLoader.dataset)


def visualize(model, source, target, args):
    source_encoder = model["Source_encoder"].eval()
    target_encoder = model["Target_encoder"].eval()
    shared_encoder = model["Shared_encoder"].eval()
    shared_decoder = model["Shared_decoder"].eval()
    batch_size = args.batch_size
    figure_path = args.figure_path

    with torch.no_grad():
        ''' Source Data '''
        source_private = source_encoder(source)
        source_shared = shared_encoder(source)

        s_out_sh_re = shared_decoder(source_private + source_shared)
        s_sh_re = shared_decoder(source_shared)
        s_out_re = shared_decoder(source_private)
        s_result = torch.cat([source, s_out_sh_re, s_sh_re, s_out_re])
        save_image(make_grid(s_result, nrow=batch_size), figure_path + "source.png")
        print("Source image saved to {}".format(figure_path + "source.png"))

        ''' Target Data '''
        target_private = target_encoder(target)
        target_shared = shared_encoder(target)
        
        t_out_sh_re = shared_decoder(target_private + target_shared)
        t_sh_re = shared_decoder(target_shared)
        t_out_re = shared_decoder(target_private)
        t_result = torch.cat([target, t_out_sh_re, t_sh_re, t_out_re])
        save_image(make_grid(t_result, nrow=batch_size), figure_path + "target.png")
        print("Target image saved to {}".format(figure_path + "target.png"))
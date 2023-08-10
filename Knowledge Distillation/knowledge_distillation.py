import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import *
from data_loader import DataLoader
from models.simplenet import SimpleNet
from models.resnet import ResNet18

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Implementation of knowledge distillation')
parser.add_argument('-m', '--mixup', default=True, type=bool)
parser.add_argument('-t', '--temperature', default=4, type=int)
parser.add_argument('-r', '--rate', default=0.9, type=float)
args = parser.parse_args()

device = getDevice()

writer = SummaryWriter('./runs/FMNIST/T{}_R{}'.format(args.temperature, args.rate))

teacher = ResNet18()
teacher.load_state_dict(torch.load('./checkpoint/teacher18_FMNIST_cpkt')['model'])
teacher = teacher.to(device)

student = SimpleNet().to(device)

num_epochs = 100
lr = 0.01
batch_size = 128
# optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=5e-4)
optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

train_loader, val_loader, test_loader = DataLoader(batch_size=batch_size, train_val_split=0.8, mixup=args.mixup)
train_total_loss, train_total_acc, val_total_loss, val_total_acc = [], [], [], []

best_acc = 0.0

state = {
    'model': [],
    'acc': 0,
    'epoch': 0,
}

print('----- Start Training ----- T: {}, R: {}'.format(args.temperature, args.rate))
start = time.process_time()
for epoch in range(num_epochs):
    student.train()
    train_total = 0
    train_loss, train_correct = 0, 0
    for x, label in train_loader:
        optimizer.zero_grad()

        x, label = x.to(device), label.to(device)
        with torch.no_grad():
            soft_label = teacher(x)
        output = student(x)
        
        _, predicted = torch.max(output.data, 1)
        train_total += label.size(0)

        label = torch.argmax(label.data, 1)
        train_correct += (predicted == label).sum().item()
        
        loss = loss_fn_kd(output, label, soft_label, args.temperature, args.rate)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_total_loss.append(train_loss / len(train_loader))
    train_total_acc.append(100 * train_correct / train_total)

    val_total = 0
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        student.eval()
        for data, labels in val_loader:
            data, target = data.to(device), labels.to(device)
            outputs = student(data)
            val_loss += loss_fn(outputs, target).item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

        val_total_loss.append(val_loss / len(val_loader))
        val_total_acc.append(100 * val_correct / val_total)

    scheduler.step()

    print('Epoch: {}/{}'.format(epoch+1, num_epochs))
    print('[Train] loss: {:.5f}, acc: {:.2f}%'.format(train_total_loss[-1], train_total_acc[-1]))
    print('[Val]   loss: {:.5f}, acc: {:.2f}%'.format(val_total_loss[-1], val_total_acc[-1]))

    writer.add_scalar('Loss/train', train_total_loss[-1], epoch)
    writer.add_scalar('Accuracy/train', train_total_acc[-1], epoch)
    writer.add_scalar('Loss/val', val_total_loss[-1], epoch)
    writer.add_scalar('Accuracy/val', val_total_acc[-1], epoch)

    # save checkpoint
    if val_total_acc[-1] > best_acc:
        best_acc = val_total_acc[-1]
        state['model'] = student.state_dict()
        state['acc'] = val_total_acc[-1]
        state['epoch'] = epoch
        print('- New checkpoint -')

print(f'----- Traing time: {time.process_time() - start} s -----')

print('\nLoading best model...')
student.load_state_dict(state['model'])
print('Best acc: {}%'.format(state['acc']))

with torch.no_grad():
    total, correct = 0, 0
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = student(data)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
    print(f'\n[Test] Accuracy: {100 * correct / total}%')

show_train_result(num_epochs, train_total_loss, train_total_acc, val_total_loss, val_total_acc, 'student_kd_T{}_R{}'.format(args.temperature, args.rate))
import os
import time
import argparse
import torch
from data_loader import DataLoader
from models.simplenet import SimpleNet
from utils import *

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Train student network')
parser.add_argument('-m', '--mixup', default=False, type=bool)
args = parser.parse_args()

device = getDevice()
model = SimpleNet().to(device)

num_epochs = 1000
lr = 0.001
batch_size = 128
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

train_loader, val_loader, test_loader = DataLoader(batch_size=batch_size, train_val_split=0.8, mixup=args.mixup)
train_total_loss, train_total_acc, val_total_loss, val_total_acc = [], [], [], []

best_acc = 0.0
print('\n----- start training -----')
start = time.process_time()
for epoch in range(num_epochs):
    model.train()
    train_total = 0
    train_loss, train_correct = 0, 0
    for idx, (x, label) in enumerate(train_loader):
        x, label = x.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(x)

        predicted = torch.argmax(output.data, 1)
        train_total += label.size(0)

        label = torch.argmax(label.data, 1)
        train_correct += (predicted == label).sum().item()

        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_total_loss.append(train_loss / len(train_loader))
    train_total_acc.append(100 * train_correct / train_total)

    val_total = 0
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            predicted = torch.argmax(outputs.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

            val_loss += loss_fn(outputs, target).item()
    val_total_loss.append(val_loss / len(val_loader))
    val_total_acc.append(100 * val_correct / val_total)

    scheduler.step()
    
    print('Epoch: {}/{}'.format(epoch+1, num_epochs))
    print('[Train] loss: {:.5f}, acc: {:.2f}%'.format(train_total_loss[-1], train_total_acc[-1]))
    print('[Val]   loss: {:.5f}, acc: {:.2f}%'.format(val_total_loss[-1], val_total_acc[-1]))

     # save checkpoint
    if val_total_acc[-1] > best_acc:
        state = {
            'model': model.state_dict(),
            'acc': val_total_acc[-1],
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/student_cpkt')
        best_acc = val_total_acc[-1]
        print('- New checkpoint -')
print(f'Traing time: {time.process_time() - start} s')

print('\nLoading best model...')
checkpoint = torch.load('./checkpoint/student_cpkt')
model.load_state_dict(checkpoint['model'])
print('Best acc: {}%'.format(checkpoint['acc']))

# test on testing data
with torch.no_grad():
    total, correct = 0, 0
    for data, labels in test_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)

        predicted = torch.argmax(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
    print(f'\n[Test] Accuracy: {100 * correct / total}%')

show_train_result(num_epochs, train_total_loss, train_total_acc, val_total_loss, val_total_acc, 'SimpleNet')

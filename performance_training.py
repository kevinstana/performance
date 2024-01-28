import time
import glob
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


class MLProject2Dataset(Dataset):
    def __init__(self, data_dir, metadata_fname='metadata.csv',
                 transform=None):
        self.data_dir = data_dir
        self.transform = transform

        image_paths = glob.glob(self.data_dir + 'images/*')
        image_data = []
        for path in image_paths:
            filename = path.split('/')[3].split('.')[0]
            # filename = path.split('\\')[1].split('.')[0]
            image_data.append((filename, path))
        id_path_columns = ['image_id', 'path']
        id_path_df = pd.DataFrame(image_data, columns=id_path_columns)

        csv_path = self.data_dir + metadata_fname
        metadata_columns = ['image_id', 'dx']
        metadata_df = pd.read_csv(csv_path, usecols=metadata_columns)
        metadata_df['dx'] = pd.Categorical(metadata_df['dx'])
        metadata_df['dx'] = metadata_df['dx'].cat.codes

        merge_column = 'image_id'
        self.merged_df = pd.merge(id_path_df, metadata_df, on=merge_column)

        self.dataset_size = self.merged_df.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        row = self.merged_df.iloc[idx]
        path = row.path
        label = row.dx
        image = torchvision.io.read_image(path).float()

        if self.transform is not None:
            image = self.transform(image)
        image = image / 255.0

        return image, label
    

def monitor_CPU_Ram():
    mem = psutil.virtual_memory()
    date = time.time()
    ram = mem.percent
    usage_per_cpu = psutil.cpu_percent(interval=1.0, percpu=True)
    return date, ram, usage_per_cpu

def train_net(
        model: nn.Module, trainloader: DataLoader, epochs: int = 10,
        optimizer: optim = None, loss: nn.modules.loss = None,
        device: str = 'cpu', print_period: int = 10, model_name = None) -> None:

    total = 0
    correct = 0
    running_loss = 0.0
    model.train()
    for epoch in range(epochs):
        start = time.time()
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            current_loss = loss(pred, y.long())
            current_loss.backward()
            optimizer.step()
            running_loss += current_loss.item()
            yhat = torch.argmax(pred, 1)
            total += y.size(0)
            correct += (yhat == y).type(torch.float).sum().item()

        date, ram, cpu = monitor_CPU_Ram()
        duration = date - start
        avg_loss = running_loss / len(trainloader)
        accuracy = correct / total

        if model_name is not None:
            res18_times.append(duration)
            res18_accuracy.append(accuracy)
            for i, cpu_percent in enumerate(cpu, 0):
                res18_cpu[i].append(cpu_percent)
            res18_memory.append(ram)
        else:
            res34_times.append(duration)
            res34_accuracy.append(accuracy)
            for i, cpu_percent in enumerate(cpu, 0):
                res34_cpu[i].append(cpu_percent)
            res34_memory.append(ram)

        print(f'[Epoch {epoch}], Duration {duration}')
        print(f'{total} Images, AvgLoss: {avg_loss}, Accuracy: {accuracy}')
        print(f'DATE: {time.ctime(date)} | MEMORY: {ram} | CPU: {cpu}')

        total = 0
        correct = 0
        running_loss = 0.0


# def test_net(
#         model: nn.Module, testloader: DataLoader,
#         loss: nn.modules.loss = None, device: str = 'cpu') -> None:

#     test_total = 0
#     test_correct = 0
#     test_loss = 0.0
#     model.eval()
#     with torch.no_grad():
#         for (X, y) in testloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss(pred, y.long()).item()
#             yhat = torch.argmax(pred, 1)
#             test_total += y.size(0)
#             test_correct += (yhat == y).type(torch.float).sum().item()

#     avg_test_loss = test_loss / len(testloader)
#     test_accuracy = test_correct / test_total
#     return test_total, avg_test_loss, test_accuracy


train_comp = [
    transforms.RandomResizedCrop(224, antialias=False),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

# test_comp = [
#     transforms.Resize(256, antialias=False),
#     transforms.CenterCrop(224),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]

train_transform = transforms.Compose(train_comp)
# test_transform = transforms.Compose(test_comp)

data_dir = './dermoscopy_classification/'
csv = 'metadata.csv'
generator = torch.Generator().manual_seed(42)
lengths = [200, 9800]

dermoscopy_dataset_1 = MLProject2Dataset(data_dir, csv, train_transform)
training_set, _ = random_split(dermoscopy_dataset_1, lengths, generator)

# dermoscopy_dataset_2 = MLProject2Dataset(data_dir, csv, test_transform)
# _, big_testing_set = random_split(dermoscopy_dataset_2, lengths, generator)

# lengths = [90] * 100
# test_sets = random_split(big_testing_set, lengths, generator)

device = ("cpu")
loss = nn.CrossEntropyLoss()
epochs = 5

mem = psutil.virtual_memory()
cores = psutil.cpu_count()
res18_cpu = [[] for core in range(cores)]
res34_cpu = [[] for core in range(cores)]
res18_memory = []
res34_memory = []
res18_times = []
res34_times = []
res18_accuracy = []
res34_accuracy = []

print(f'Nuber of CPUs: {cores}, Total physical memory: {str(int(mem.total/1024**2))} MB')

full_test_start = time.time()
print()
print('**********RESNET 18**********')
trainloader = DataLoader(training_set, batch_size=32, shuffle=True)
lr = 1e-1
net = torchvision.models.resnet18(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
print(f'TRAIN, lr={lr}')
start = time.time()
train_net(net, trainloader, epochs, optimizer, loss, device, model_name=18)
res18_lr1_duration = time.time() - start
print(f'Duration: {res18_lr1_duration}')

lr = 1e-2
net = torchvision.models.resnet18(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
print()
print(f'TRAIN, lr={lr}')
start = time.time()
train_net(net, trainloader, epochs, optimizer, loss, device, model_name=18)
res18_lr2_duration = time.time() - start
print(f'Duration: {res18_lr2_duration}')

lr = 1e-3
net = torchvision.models.resnet18(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
print()
print(f'TRAIN, lr={lr}')
start = time.time()
train_net(net, trainloader, epochs, optimizer, loss, device, model_name=18)
res18_lr3_duration = time.time() - start
print(f'Duration: {res18_lr3_duration}')

# print()
# print('TEST')
# for batch, test in enumerate(test_sets, 0): 
#     testloader = DataLoader(test, batch_size=32, shuffle=True)
#     start = time.time()
#     total, avg_loss, accuracy = test_net(net, testloader, loss, device)
#     date, ram, usage_per_cpu = monitor_CPU_Ram()
#     duration = date - start
#     res18_times.append(duration)
#     res18_accuracy.append(accuracy)
#     print(f'[Batch {batch}], Duration: {duration}')
#     print(f'{total} Images, AvgLoss: {avg_loss}, Acc: {accuracy}')
#     print(f'DATE: {time.ctime(date)} | MEMORY: {ram} | CPU: {usage_per_cpu}')
#     for i, cpu in enumerate(usage_per_cpu, 0):
#         res18_cpu_percent_list[i].append(cpu)
#     res18_memory.append(ram)
 
print()
print()
print('**********RESNET 34**********')
lr = 1e-1
net = torchvision.models.resnet34(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
print(f'TRAIN, lr={lr}')
start = time.time()
train_net(net, trainloader, epochs, optimizer, loss, device)
res34_lr1_duration = time.time() - start
print(f'Duration: {res34_lr1_duration}')

lr = 1e-2
net = torchvision.models.resnet34(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
print()
print(f'TRAIN, lr={lr}')
start = time.time()
train_net(net, trainloader, epochs, optimizer, loss, device)
res34_lr2_duration = time.time() - start
print(f'Duration: {res34_lr2_duration}')

lr = 1e-3
net = torchvision.models.resnet34(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
print()
print(f'TRAIN, lr={lr}')
start = time.time()
train_net(net, trainloader, epochs, optimizer, loss, device)
res34_lr3_duration = time.time() - start
print(f'Duration: {res34_lr3_duration}')

# print()
# print('TEST')
# for batch, test in enumerate(test_sets, 0): 
#     testloader = DataLoader(test, batch_size=32, shuffle=True)
#     start = time.time()
#     total, avg_loss, accuracy = test_net(net, testloader, loss, device)
#     date, ram, usage_per_cpu = monitor_CPU_Ram()
#     duration = date - start
#     res34_times.append(duration)
#     res34_accuracy.append(accuracy)
#     print(f'[Batch {batch}], Duration: {duration}')
#     print(f'{total} Images, AvgLoss: {avg_loss}, Acc: {accuracy}')
#     print(f'DATE: {time.ctime(date)} | MEMORY: {ram} | CPU: {usage_per_cpu}')
#     for i, cpu in enumerate(usage_per_cpu, 0):
#         res34_cpu_percent_list[i].append(cpu)
#     res34_memory.append(ram)

full_test_duration = time.time() - full_test_start
print()
print(f'Full duration: {full_test_duration}')

print()
print('FINAL RESULTS')
print('[RESNET 18]')
print(f'lr=1e-1, Time: {res18_lr1_duration}, Accuracy: {res18_accuracy[4]}')
print(f'lr=1e-2, Time: {res18_lr2_duration}, Accuracy: {res18_accuracy[9]}')
print(f'lr=1e-3, Time: {res18_lr3_duration}, Accuracy: {res18_accuracy[14]}')
print()
print('[RESNET 34]')
print(f'lr=1e-1, Time: {res34_lr1_duration}, Accuracy: {res34_accuracy[4]}')
print(f'lr=1e-2, Time: {res34_lr2_duration}, Accuracy: {res34_accuracy[9]}')
print(f'lr=1e-3, Time: {res34_lr3_duration}, Accuracy: {res34_accuracy[14]}')

# Resnet18 cpus
plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(5), res18_cpu[i][:5], label=f'Core {i + 1}', marker=',')
plt.xlabel('Epochs')
plt.ylabel('CPU Utilization')
plt.title('Resnet18 Training, CPU Utilization for lr=1e-1')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(5), res18_cpu[i][5:10], label=f'Core {i + 1}', marker=',')
plt.xlabel('Epochs')
plt.ylabel('CPU Utilization')
plt.title('Resnet18 Training, CPU Utilization for lr=1e-2')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(5), res18_cpu[i][10:], label=f'Core {i + 1}', marker=',')
plt.xlabel('Epochs')
plt.ylabel('CPU Utilization')
plt.title('Resnet18 Training, CPU Utilization for lr=1e-3')
plt.legend()
plt.grid(True)

# Resnet18 memory
plt.figure(figsize=(12, 6))
plt.plot(range(5), res18_memory[:5], label='lr=1e-1', marker=',')
plt.plot(range(5), res18_memory[5:10], label='lr=1e-2', marker=',')
plt.plot(range(5), res18_memory[10:], label='lr=1e-3', marker=',')
plt.xlabel('Epochs')
plt.ylabel('Memory Utilization')
plt.title('Resnet18 Training, Memory Utilization')
plt.legend()
plt.grid(True)

# Resnet18 accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(5), res18_accuracy[:5], label='lr=1e-1', marker=',')
plt.plot(range(5), res18_accuracy[5:10], label='lr=1e-2', marker=',')
plt.plot(range(5), res18_accuracy[10:], label='lr=1e-3', marker=',')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Resnet18 Training, Accuracy')
plt.legend()
plt.grid(True)

# Resnet18 times
plt.figure(figsize=(12, 6))
plt.plot(range(5), res18_times[:5], label='lr=1e-1', marker=',')
plt.plot(range(5), res18_times[5:10], label='lr=1e-2', marker=',')
plt.plot(range(5), res18_times[10:], label='lr=1e-3', marker=',')
plt.xlabel('Epochs')
plt.ylabel('Time')
plt.title('Resnet18 Training, Time')
plt.legend()
plt.grid(True)

# Resnet34 cpus
plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(5), res34_cpu[i][:5], label=f'Core {i + 1}', marker=',')
plt.xlabel('Epochs')
plt.ylabel('CPU Utilization')
plt.title('Resnet34 Training, CPU Utilization for lr=1e-1')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(5), res34_cpu[i][5:10], label=f'Core {i + 1}', marker=',')
plt.xlabel('Epochs')
plt.ylabel('CPU Utilization')
plt.title('Resnet34 Training, CPU Utilization for lr=1e-2')
plt.legend()
plt.grid(True)

plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(5), res34_cpu[i][10:], label=f'Core {i + 1}', marker=',')
plt.xlabel('Epochs')
plt.ylabel('CPU Utilization')
plt.title('Resnet34 Training, CPU Utilization for lr=1e-3')
plt.legend()
plt.grid(True)

# Resnet34 memory
plt.figure(figsize=(12, 6))
plt.plot(range(5), res34_memory[:5], label='lr=1e-1', marker=',')
plt.plot(range(5), res34_memory[5:10], label='lr=1e-2', marker=',')
plt.plot(range(5), res34_memory[10:], label='lr=1e-3', marker=',')
plt.xlabel('Epochs')
plt.ylabel('Memory Utilization')
plt.title('Resnet34 Training, Memory Utilization')
plt.legend()
plt.grid(True)

# Resnet34 accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(5), res34_accuracy[:5], label='lr=1e-1', marker=',')
plt.plot(range(5), res34_accuracy[5:10], label='lr=1e-2', marker=',')
plt.plot(range(5), res34_accuracy[10:], label='lr=1e-3', marker=',')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Resnet34 Training, Accuracy')
plt.legend()
plt.grid(True)

# Resnet34 times
plt.figure(figsize=(12, 6))
plt.plot(range(5), res34_times[:5], label='lr=1e-1', marker=',')
plt.plot(range(5), res34_times[5:10], label='lr=1e-2', marker=',')
plt.plot(range(5), res34_times[10:], label='lr=1e-3', marker=',')
plt.xlabel('Epochs')
plt.ylabel('Time')
plt.title('Resnet34 Training, Times')
plt.legend()
plt.grid(True)

plt.show()

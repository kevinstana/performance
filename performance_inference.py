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
        device: str = 'cpu', print_period: int = 10) -> None:

    total = 0
    correct = 0
    running_loss = 0.0
    model.train()
    for epoch in range(epochs):
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

        avg_loss = running_loss / len(trainloader)
        accuracy = correct / total
        print(f'[Epoch {epoch}] {total} Images, Average Loss: {avg_loss}, Accuracy: {accuracy}')

        total = 0
        correct = 0
        running_loss = 0.0


def test_net(
        model: nn.Module, testloader: DataLoader,
        loss: nn.modules.loss = None, device: str = 'cpu') -> None:

    test_total = 0
    test_correct = 0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for (X, y) in testloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss(pred, y.long()).item()
            yhat = torch.argmax(pred, 1)
            test_total += y.size(0)
            test_correct += (yhat == y).type(torch.float).sum().item()

    avg_test_loss = test_loss / len(testloader)
    test_accuracy = test_correct / test_total
    return test_total, avg_test_loss, test_accuracy


train_comp = [
    transforms.RandomResizedCrop(224, antialias=False),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

test_comp = [
    transforms.Resize(256, antialias=False),
    transforms.CenterCrop(224),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

train_transform = transforms.Compose(train_comp)
test_transform = transforms.Compose(test_comp)

data_dir = './dermoscopy_classification/'
csv = 'metadata.csv'
generator = torch.Generator().manual_seed(42)
lengths = [0.1, 0.9]

dermoscopy_dataset_1 = MLProject2Dataset(data_dir, csv, train_transform)
training_set, _ = random_split(dermoscopy_dataset_1, lengths, generator)

dermoscopy_dataset_2 = MLProject2Dataset(data_dir, csv, test_transform)
_, big_testing_set = random_split(dermoscopy_dataset_2, lengths, generator)

lengths = [90] * 100
test_sets = random_split(big_testing_set, lengths, generator)

device = ("cpu")
loss = nn.CrossEntropyLoss()
epochs = 5

res18_times = []
res34_times = []
res18_accuracy = []
res34_accuracy = []

mem = psutil.virtual_memory()
cores = psutil.cpu_count()
res18_cpu_percent_list = [[] for core in range(cores)]
res34_cpu_percent_list = [[] for core in range(cores)]
res18_memory = []
res34_memory = []

print(f'Nuber of CPUs: {cores}, Total physical memory: {str(int(mem.total/1024**2))} MB')

full_test_start = time.time()
print()
print('**********RESNET 18**********')
net = torchvision.models.resnet18(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
trainloader = DataLoader(training_set, batch_size=32, shuffle=True)
print('TRAIN')
train_net(net, trainloader, epochs, optimizer, loss, device)

print()
print('TEST')
for batch, test in enumerate(test_sets, 0): 
    testloader = DataLoader(test, batch_size=32, shuffle=True)
    start = time.time()
    total, avg_loss, accuracy = test_net(net, testloader, loss, device)
    date, ram, usage_per_cpu = monitor_CPU_Ram()
    duration = date - start
    res18_times.append(duration)
    res18_accuracy.append(accuracy)
    print(f'[Batch {batch}], Duration: {duration}')
    print(f'{total} Images, AvgLoss: {avg_loss}, Acc: {accuracy}')
    print(f'DATE: {time.ctime(date)} | MEMORY: {ram} | CPU: {usage_per_cpu}')
    for i, cpu in enumerate(usage_per_cpu, 0):
        res18_cpu_percent_list[i].append(cpu)
    res18_memory.append(ram)
 
print()
print('**********RESNET 34**********')
net = torchvision.models.resnet34(weights='DEFAULT').to(device)
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
print('TRAIN')
train_net(net, trainloader, epochs, optimizer, loss, device)

print()
print('TEST')
for batch, test in enumerate(test_sets, 0): 
    testloader = DataLoader(test, batch_size=32, shuffle=True)
    start = time.time()
    total, avg_loss, accuracy = test_net(net, testloader, loss, device)
    date, ram, usage_per_cpu = monitor_CPU_Ram()
    duration = date - start
    res34_times.append(duration)
    res34_accuracy.append(accuracy)
    print(f'[Batch {batch}], Duration: {duration}')
    print(f'{total} Images, AvgLoss: {avg_loss}, Acc: {accuracy}')
    print(f'DATE: {time.ctime(date)} | MEMORY: {ram} | CPU: {usage_per_cpu}')
    for i, cpu in enumerate(usage_per_cpu, 0):
        res34_cpu_percent_list[i].append(cpu)
    res34_memory.append(ram)

full_test_duration = time.time() - full_test_start
print()
print(f'Full duration: {full_test_duration}')

print()
print('FINAL RESULTS')
print('[RESNET 18]')
print('Max time:', max(res18_times))
print('Min time:', min(res18_times))
print('Mean time:', np.mean(res18_times))
print('Time standard deviation:', np.std(res18_times))
print('Max accuracy:', max(res18_accuracy))
print('Min accuracy:', min(res18_accuracy))
print('Mean accuracy:', np.mean(res18_accuracy))
print('Accuracy standard deviation:', np.std(res18_accuracy))
print('Tail latency (99P):', np.percentile(res18_times, 99))
print()
print('[RESNET 34]')
print('Max time:', max(res34_times))
print('Min time:', min(res34_times))
print('Mean time:', np.mean(res34_times))
print('Time standard deviation:', np.std(res34_times))
print('Max accuracy:', max(res34_accuracy))
print('Min accuracy:', min(res34_accuracy))
print('Mean accuracy:', np.mean(res34_accuracy))
print('Accuracy standard deviation:', np.std(res34_accuracy))
print('Tail latency (99P):', np.percentile(res34_times, 99))

# Resnet18 cpus
plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(100), res18_cpu_percent_list[i], label=f'Core {i + 1}', marker=',')
plt.xlabel('Batches')
plt.ylabel('CPU Utilization')
plt.title('Resnet18 Testing, CPU Utilization For 100 Batches')
plt.legend()
plt.grid(True)

# # Resnet34 cpus
plt.figure(figsize=(12, 6))
for i in range(cores):
    plt.plot(range(100), res34_cpu_percent_list[i], label=f'Core {i + 1}', marker=',')
plt.xlabel('Batches')
plt.ylabel('CPU Utilization')
plt.title('Resnet34 Testing, CPU Utilization For 100 Batches')
plt.legend()
plt.grid(True)

# Memory
plt.figure(figsize=(12, 6))
plt.plot(range(100), res18_memory, label='Resnet18', marker=',')
plt.plot(range(100), res34_memory, label='Resnet34', marker=',')
plt.xlabel('Batches')
plt.ylabel('Memory Utilization')
plt.title('Testing, Memory Utilization For 100 Batches')
plt.legend()
plt.grid(True)

# Accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(100), res18_accuracy, label='Resnet18', marker=',')
plt.plot(range(100), res34_accuracy, label='Resnet34', marker=',')
plt.xlabel('Batches')
plt.ylabel('Accuracy')
plt.title('Testing, Accuracy For 100 Batches')
plt.legend()
plt.grid(True)

# Times
plt.figure(figsize=(12, 6))
plt.plot(range(100), res18_times, label='Resnet18', marker=',')
plt.plot(range(100), res34_times, label='Resnet34', marker=',')
plt.xlabel('Batches')
plt.ylabel('Time')
plt.title('Testing, Times and batches')
plt.legend()
plt.grid(True)

plt.show()

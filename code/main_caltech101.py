import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar
from tensor_sampling import *
from vector_sampling import *
import os
from torchinfo import summary
from pytorch_pretrained_vit import ViT
from torcheval.metrics.functional import multiclass_auroc


parser = argparse.ArgumentParser(description='PyTorch MCL Training')
parser.add_argument('--mode', default='tensor',
                    help='mode for sampling')

args = parser.parse_args()
best_acc = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
])

data_dir = '/home/kimishima/pytorch-classification-uncertainty/data/caltech101/101_ObjectCategories'
dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

dataset_size = len(dataset)
train_ratio = 0.7
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 128, shuffle = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 128)
num_class = 101
input_channel = 3
model = ViT('B_32_imagenet1k', pretrained=True, in_channels = input_channel, image_size = 384, num_classes = num_class)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0,1,2])
model = model.to(device)

def train(epoch, num_class, TensorSensing, VectorSensing):
    print("Epoch:", epoch+1)
    model.train()
    criterion = nn.CrossEntropyLoss()
    train_loss = 0
    correct = 0
    total = 0
    auc_score = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if TensorSensing == False:
            inputs = inputs
        else:
            inputs = TensorSensing(inputs)
        if VectorSensing == False:
            inputs = inputs
        else:
            inputs = VectorSensing(inputs)
        output = model(inputs)
        loss = criterion(output, targets)
        auc_score += multiclass_auroc(input = output, target = targets, num_classes = num_class, average = "macro").item() * 100
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, num_class, current_weight, TensorSensing, VectorSensing):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    auc_score = 0
    cnt = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                if TensorSensing == False:
                    inputs = inputs
                else:
                    inputs = TensorSensing(inputs)
                if VectorSensing == False:
                    inputs = inputs
                else:
                    inputs = VectorSensing(inputs)
                output = model(inputs)
                loss = criterion(output, targets)
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                cnt += 1
                auc_score += multiclass_auroc(input = output, target = targets, num_classes = num_class, average = "macro").item() * 100
                progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    total_auc = auc_score/cnt
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'auc': total_auc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if TensorSensing == False and VectorSensing  == False:
            torch.save(model.state_dict(), './checkpoint/pretrain_caltech101.pth')
        torch.save(state, './checkpoint/MCL_caltech101.pickle')
        current_weight = True
        best_acc = acc
    print("Best Accuracy:", best_acc)

current_directory = os.getcwd()
image_folder_path = os.path.join(current_directory, 'checkpoint')
sample_file_path = os.path.join(image_folder_path, 'pretrain_caltech101.pth')
if not os.path.exists(image_folder_path) or not os.path.exists(sample_file_path):
    print("Make the pre-train model.")
    #pre-trainモデルを作成する
    LR = [1e-3, 1e-3, 1e-4, 1e-5]
    Epoch = [40,40,40,40]
    current_weight = False
    for lr, epoch in zip(LR, Epoch):
        optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=0.9, weight_decay=0.0001)
        if current_weight:
            model.load_state_dict(torch.load('./checkpoint/pretrain_caltech101.pth'))
        for iteration in range(epoch):
            train(iteration, num_class, False, False)
            test(iteration, num_class, current_weight, False, False)

model.load_state_dict(torch.load("./checkpoint/pretrain_caltech101.pth"))
best_acc = 0
optimizer = optim.SGD(model.parameters(), lr=1e-4,
                        momentum=0.9, weight_decay=0.0001)
current_weight = False

if args.mode == "tensor":
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        W1, W2, W3 = HOSVD(inputs, 163, 163, 1, device)#(334,334,1), (235, 235, 1), (163, 163, 1)
        break
    TensorSensing = TensorSensing((163, 163, 1), W1, W2, W3, W1.T, W2.T, W3.T, linear_sensing=False, separate_decoder=True, regularizer=None, constraint=None, device=device)
    VectorSensing = False
elif args.mode == "vector":
    for inputs, labels in train_loader:
        sample = inputs.to(device)
        sample_R = sample[:,0,:,:]
        sample_G = sample[:,1,:,:]
        sample_B = sample[:,2,:,:]
        height = 8
        R_p = PCA(sample_R, height, device)
        G_p = PCA(sample_G, height, device)
        B_p = PCA(sample_B, height, device)
        projection = torch.stack([R_p, G_p, B_p], dim = 1)
        break
    TensorSensing = False
    VectorSensing = VectorSensing(96, projection)

for epoch in range(100):
    train(epoch, num_class, TensorSensing, VectorSensing)
    test(epoch, num_class, current_weight, TensorSensing, VectorSensing)
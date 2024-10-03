import torch
from torch import nn
from torch.utils import data
from torch.utils.data import Dataset
from torchvision import transforms,datasets,io
import time
import numpy as np
import os
import pandas as pd


def load_data_mnist(batch_size,resize=None):
    
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train=datasets.FashionMNIST(root='./data',train=True,transform=trans,download=True)
    mnist_test=datasets.FashionMNIST(root='./data',train=False,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=8,pin_memory=True),
            data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=8,pin_memory=True))
def load_data_cifar10(batch_size,resize=None,transform=False):

    if transform:
        trans=[
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]     
    else:
        trans=[transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=datasets.CIFAR10(root='./cifar',train=True,transform=trans,download=True)
    mnist_test=datasets.CIFAR10(root='./cifar',train=False,transform=trans,download=True)
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=8,pin_memory=True),
            data.DataLoader(mnist_test,batch_size,shuffle=True,num_workers=8,pin_memory=True))                      

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.
    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
                
data_transform={
    'train':transforms.Compose([ 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])   
    ]),
    'test':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])   
    ])
}
def mydataloader(root,csvfile,batch_size,transform=data_transform['train']):
    full=CustomDataset(root,csvfile,transform=transform)
    train_size=int(0.8*len(full))
    val_size=len(full)-train_size
    train,val=data.random_split(full,(train_size,val_size))
    
    return  (data.DataLoader(train,batch_size,shuffle=True,num_workers=8,pin_memory=True),
            data.DataLoader(val,batch_size,shuffle=True,num_workers=8,pin_memory=True))

class CustomDataset(Dataset):
    def __init__(self,root,csv_file,transform=None,target_transform=None):
        self.df=pd.read_csv(os.path.join(root,csv_file))
        self.img_dir=os.path.join(root,'train')
        self.transform=transform
        self.target_transform=target_transform
        self.annot_dict = {}
        for i, lbl in enumerate(self.df['label'].unique()):
            self.annot_dict[lbl] = i

      # {lbl: class}
        self.annot_dict_reversed = {v:k for k,v in self.annot_dict.items()}

      # add a column with decoded labels
        self.df['encoded'] = self.df['label'].map(self.annot_dict)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index) :
        img_path=os.path.join(self.img_dir,str(self.df.iloc[index,0])+'.png')
        image=io.read_image(img_path)
        label=self.df.iloc[index,2]
        if self.transform:
            image=self.transform(image)
        return image,label


def accuracy(y_hat, y):  

    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)#最大概率的分类
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')  # load一本书







def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
            
    return metric[0] / metric[1]


def train(net, train_iter, val_iter, num_epochs, lr, 
              device,premodel=0):
    """Train and evaluate a model with CPU or GPU."""    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    if premodel==0:
        net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler=torch.optim.lr_scheduler.StepLR(optimizer,lr_period,lr_decay)
    loss = nn.CrossEntropyLoss()
    timer = Timer()
    for epoch in range(num_epochs):
        metric = Accumulator(3)  # train_loss, train_acc, num_examples
        net.train()    
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device) 
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0],accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0]/metric[2], metric[1]/metric[2]
        # scheduler.step()    
        test_acc = evaluate_accuracy_gpu(net, val_iter)    
        print('loss %.3f, train acc %.3f, test acc %.3f' % (
            train_loss, train_acc, test_acc))
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
        
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')   


class Accumulator:  #@save
    """Sum a list of numbers over time."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
def shape(net):
    X=torch.rand(size=(1,3,224,224))
    for layer in net :
        X=layer(X)
        print(layer.__class__.__name__,'output shape: ',X.shape)

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
    
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
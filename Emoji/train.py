import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import Dataset
import torchvision
import os
from torch.utils.data import random_split
cur_path=os.path.dirname(__file__)
data_dir=os.path.join(cur_path,'data')
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
min_loss = float('inf')
# Hyper parameters
num_epochs = 15
num_classes = 4
batch_size = 128
learning_rate = 0.0015
model_path=os.path.join(cur_path,'model.ckpt')
n_epochs_stop = 3  # Number of epochs after which to stop if no improvement
best_val_loss = float("Inf") 

class EmojiDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.images = []
        for class_ in self.classes:
            class_dir = os.path.join(root_dir, class_)
            for image_name in os.listdir(class_dir):
                self.images.append((os.path.join(class_dir, image_name), self.classes.index(class_)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image= Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
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
# Create the dataset
dataset = EmojiDataset(data_dir)
# Determine the lengths of the splits
lengths = [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
# Create the train and validation splits
train_dataset, test_dataset = random_split(dataset, lengths)
train_dataset.dataset.transform = data_transform['train'] 
test_dataset.dataset.transform = data_transform['test']

# Data loader
train_loader =data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader =data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)

def block(input_channels, num_channels, num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels,num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return nn.Sequential(*blk)
def shape(net):
    X=torch.rand(size=(1,3,224,224))
    for layer in net :
        X=layer(X)
        print(layer.__class__.__name__,'output shape: ',X.shape)
b1= nn.Sequential(
            nn.Conv2d(3,64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2=block(64,64,2,first_block=True)
b3=block(64,128,2)
b4=block(128,256,2)
b5=block(256,512,2)
net=nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,num_classes))

# shape(net)
model = net.to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
if __name__ == '__main__':
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = criterion(outputs, labels)
            
            if (i + 1) % 20 == 0:
                if loss < min_loss:
                    torch.save(model.state_dict(), model_path)
                    min_loss = loss
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))         
        # Test the model
        val_loss=0
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                # Accumulate loss
                val_loss += batch_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        if val_loss <  best_val_loss:
            # If loss improved, save the model and reset the counter
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            # If loss did not improve, increment the counter
            epochs_no_improve += 1  
        print(best_val_loss)    
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break   
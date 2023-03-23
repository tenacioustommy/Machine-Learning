import torch
from torch import nn
from PIL import Image
from torchvision import transforms,datasets 
import tkinter as tk
from tkinter import filedialog

'''打开选择文件夹对话框'''
root = tk.Tk()
root.withdraw()
print('open your image folder') #获得选择好的文件夹
Filepath = filedialog.askopenfilename() #获得选择好的文件

print('Filepath:',Filepath)


class_names = ['1', '2', '3', 'RC', 'RL', 'UK'] #这个顺序很重要，要和训练时候的类名顺序一致
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(96,256,kernel_size=5,padding=2),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Conv2d(256,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),nn.Flatten(),
    nn.Linear(6400,4096),nn.ReLU(),nn.Dropout(p=0.5),
    nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(p=0.5),
    nn.Linear(4096,10),nn.ReLU()
) 
##载入模型并读取权重
model = net()
model.load_state_dict(torch.load("./data/detect_light.pt"))
model.to(device)
model.eval()
 
img_path = '/home/jwd/dataset/roi455.jpg'
 
#（1）此处为使用PIL进行测试的代码
transform_valid = transforms.Compose([
    transforms.Resize((56, 56), interpolation=2),
    transforms.ToTensor()
  ]
)
img = Image.open(img_path)
img_ = transform_valid(img).unsqueeze(0) #拓展维度
 
##（2）此处为使用opencv读取图像的测试代码，若使用opencv进行读取，将上面（1）注释掉即可。
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (56, 56))
# img_ = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)/255
 
img_ = img_.to(device)
outputs = model(img_)
 
#输出概率最大的类别
_, indices = torch.max(outputs,1)
percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
perc = percentage[int(indices)].item()
result = class_names[indices]
print('predicted:', result)
 
# 得到预测结果，并且从大到小排序
# _, indices = torch.sort(outputs, descending=True)
# 返回每个预测值的百分数
# percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
# print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:5]])
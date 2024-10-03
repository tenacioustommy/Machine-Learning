%load_ext tensorboard
     

%%capture
!pip install datasets
!pip install tokenizers
!pip install torchmetrics
     

%%capture
from google.colab import drive
drive.mount('/content/drive')
     

!git clone https://github.com/hkproj/pytorch-transformer.git
     

%%capture
%cd /content/pytorch-transformer
     

%tensorboard --logdir runs
     

!mkdir -p /content/drive/MyDrive/Models/pytorch-transformer/weights
!mkdir -p /content/drive/MyDrive/Models/pytorch-transformer/vocab
     

from config import get_config
cfg = get_config()
cfg['model_folder'] = '..//drive/MyDrive/Models/pytorch-transformer/weights'
cfg['tokenizer_file'] = '..//drive/MyDrive/Models/pytorch-transformer/vocab/tokenizer_{0}.json'
cfg['batch_size'] = 24
cfg['num_epochs'] = 100
cfg['preload'] = None

from train import train_model

train_model(cfg)
     


     
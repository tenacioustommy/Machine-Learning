import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from config import get_weights_path,get_config
from transformer import Transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import BilingualDataset
from pathlib import Path
import tqdm
def get_all_sentences(ds,lang):
    for example in ds:
        yield example['translation'][lang]
    
def build_tokenizer(config,ds,lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]"])
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer)
        tokenizer.save(config.tokenizer_dir)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds = load_dataset('opus_books', 'en-zh', split='train')
    
    tokenizer_src = build_tokenizer(config,ds,'en')
    tokenizer_tgt = build_tokenizer(config,ds,'zh')

    train_ds_size = int(len(ds)*0.9)
    val_ds_size = len(ds) - train_ds_size
    train_ds, val_ds = random_split(ds,[train_ds_size,val_ds_size])
    
    train_ds = BilingualDataset(train_ds,tokenizer_src,tokenizer_tgt,config['max_len'])
    val_ds = BilingualDataset(val_ds,tokenizer_src,tokenizer_tgt,config['max_len'])
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
    model = Transformer(vocab_src_len,vocab_tgt_len,config['d_model'],config['d_ff'],config['heads'],config['max_len'],config['num_layers'],config['dropout'])
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    Path(config['model_dir']).mkdir(exist_ok=True)
    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt = get_ds(config)
    model= get_model(config,len(tokenizer_src.get_vocab()),len(tokenizer_tgt.get_vocab())).to(device)
    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id('[PAD]')).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    initial_epoch=0
    global_step=0
    if config['preload']:
        model_filename = get_weights_path(config,config['preload'])
        print(f"Loading model weights from {model_filename}")
        state=torch.load(model_filename)
        initial_epoch=state['epoch']+1
        global_step=state['global_step']
        optimizer.load_state_dict(state['optimizer_state_dict'])
    
    for epoch in range(initial_epoch,config['num_epochs']):
        model.train()
        batch_iter=tqdm(train_dataloader,desc=f"Epoch {epoch}")
        for batch  in batch_iter:
            encoder_input,decoder_input,label,encoder_mask,decoder_mask = [t.to(device) for t in batch]
            optimizer.zero_grad()
            output = model(encoder_input,decoder_input,encoder_mask,decoder_mask)
            loss = loss_func(output.view(-1,output.size(-1)),label.view(-1))
            loss.backward()
            optimizer.step()
        
            global_step+=1
            batch_iter.set_postfix(loss=loss.item())
        model_filename = get_weights_path(config,epoch)
        torch.save({'epoch':epoch,'global_step':global_step,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()},model_filename)
        
if __name__ == '__main__':
    config = get_config()
    train_model(config)
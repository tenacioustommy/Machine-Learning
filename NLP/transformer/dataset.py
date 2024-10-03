import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,max_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len
        
        self.sos_token=torch.Tensor([tokenizer_tgt.token_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token=torch.Tensor([tokenizer_tgt.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token=torch.Tensor([tokenizer_tgt.token_to_id('[PAD]')],dtype=torch.int64)  
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        example = self.ds[idx]
        src = self.tokenizer_src.encode(example['translation']['en']).ids
        tgt = self.tokenizer_tgt.encode(example['translation']['zh']).ids
        
        enc_num_pad = self.max_len - len(src)-2
        dec_num_pad = self.max_len - len(tgt)-1
        if enc_num_pad<0 or dec_num_pad<0:
            raise ValueError("max_len is too small")
        encoder_input = torch.cat([self.sos_token,torch.Tensor(src),self.eos_token,self.pad_token.repeat(enc_num_pad)])
        decoder_input = torch.cat([self.sos_token,torch.Tensor(tgt),self.pad_token.repeat(dec_num_pad)])
        label = torch.cat([torch.Tensor(tgt),self.eos_token,self.pad_token.repeat(dec_num_pad)])
        assert encoder_input.size(0)==self.max_len
        assert decoder_input.size(0)==self.max_len
        assert label.size(0)==self.max_len
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0)
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0) & torch.tril(torch.ones(1,self.max_len,self.max_len),diagonal=True).bool()
        
        return encoder_input,decoder_input,label,encoder_mask,decoder_mask
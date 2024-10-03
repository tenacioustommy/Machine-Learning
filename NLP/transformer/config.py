def get_config():
    config = {
        'batch_size':8,
        "num_epochs":10,
        "lr":1e-4,
        'max_len':100,
        'd_model':512,
        'd_ff':1024,
        'heads':4,
        'model_dir':'weights',
        "model_name":'tmodel_',
        'num_layers':1,
        "preload":False,
        'dropout':0.1,
        'tokenizer_file':'tokenizer_{0}.json',
    }
    return config
    
def get_weights_path(config,epoch):
    return config['model_dir'] + '/' + config['model_name'] + str(epoch) + '.pth'
    
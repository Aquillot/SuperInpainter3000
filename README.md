# Steps for setup

matplotlib may cause problems, install it with
```shell
pip install matplotlib
```

## Linux
```shell
cd code
python -m venv .venv/
source .venv/bin/activate
pip install -r requirements.txt

mkdir data
mkdir models
```

### Old install command (don't use)
```shell
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126
```


## Windows
```shell
cd code
python -m venv .venv/
.venv\Scripts\activate
pip install -r requirements.txt

mkdir data
mkdir models
```



Tuto MAE: https://medium.com/@ovularslan/masked-autoencoders-mae-the-art-of-seeing-more-by-masking-most-pytorch-implementation-4566e08c66a6











Test de base
```
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 1e-4,
    'beta1': 0.5, # valeur classique pour Adam dans les GANs
    'beta2': 0.999, # valeur classique pour Adam dans les GANs
    'batch_size': 128,
    "total_epochs": 1,
    'd2glr': 1.0,                  # lr ratio D/G
    'num_workers': 16,             # nombre de "threads" pour le DataLoader
    'save_dir': base_path + "models",
    'current_model_name': 'epoch_1',

    'image_range': 'tanh',
    'adversarial_weight': 0.1,
    'hole_weight': 6.0, # poids de la loss dans la zone masquée plus important que dans la zone valide car on veut bien remplir les trous
    'valid_weight': 1.0,
    'pyramid_weight': 0.5,

    "mask_ratio" : 0.5,
    "train": True,
    "dataset_images_size": 128
}
```

Inverser hole_weight et valid_weight
```
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 1e-4,
    'beta1': 0.5, # valeur classique pour Adam dans les GANs
    'beta2': 0.999, # valeur classique pour Adam dans les GANs
    'batch_size': 128,
    "total_epochs": 1,
    'd2glr': 1.0,                  # lr ratio D/G
    'num_workers': 16,             # nombre de "threads" pour le DataLoader
    'save_dir': base_path + "models",
    'current_model_name': 'epoch_1',

    'image_range': 'tanh',
    'adversarial_weight': 0.1,
    'hole_weight': 1.0, # poids de la loss dans la zone masquée plus important que dans la zone valide car on veut bien remplir les trous
    'valid_weight': 6.0,
    'pyramid_weight': 0.5,

    "mask_ratio" : 0.5,
    "train": True,
    "dataset_images_size": 128
}
```

Test de base
```
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr': 1e-4,
    'beta1': 0.5, # valeur classique pour Adam dans les GANs
    'beta2': 0.999, # valeur classique pour Adam dans les GANs
    'batch_size': 128,
    "total_epochs": 1,
    'd2glr': 1.0,                  # lr ratio D/G
    'num_workers': 16,             # nombre de "threads" pour le DataLoader
    'save_dir': base_path + "models",
    'current_model_name': 'epoch_1',

    'image_range': 'tanh',
    'adversarial_weight': 0.1,
    'hole_weight': 6.0, # poids de la loss dans la zone masquée plus important que dans la zone valide car on veut bien remplir les trous
    'valid_weight': 1.0,
    'pyramid_weight': 0.5,

    "mask_ratio" : (0.1, 0.5),
    "train": True,
    "dataset_images_size": 128
}
```
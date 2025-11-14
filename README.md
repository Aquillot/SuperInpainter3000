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
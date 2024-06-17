Create enviroment

```
conda create -n project-x python=3.10 -y
conda activate project-x
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Run Scripts

```
python main.py
```

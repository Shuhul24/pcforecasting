Create an environment following the below command:
```python
conda create -n ENV_NAME python=3.10
```

Download the packages/dependencies via ```requirements.txt``` using the following command:
```python
pip install -r requirements.txt
```
Install the PyTorch library explicitly. 
```python
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

To install PyTorch3D library
```python
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
python setup.py install
```

After installing the neccessary packages/directories, train the model using the command,
```python
cd preet
python train.py
```

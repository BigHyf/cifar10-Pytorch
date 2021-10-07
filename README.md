# pytorch-cifar10
using Pytorch to do experiments on Cifar10 dataset
## Background
HW1 for ZJU-CST Artificial Intelligence Safety
## Directory Structure
```sh
.
├── data
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   └── test_batch
├── models
│   ├── densenet.py
│   ├── efficientnet.py
│   ├── __init__.py
│   ├── googlenet.py
│   ├── lenet.py
│   ├── mobilenet.py
│   ├── monilenev2.py
│   ├── model.py
│   ├── resnet.py
│   └── vgg.py
├── utils
│   └── log.py
├── images
│   ├── train_accuracy.png
│   └── train_loss.png
├── log/
├── scripts/
├── state/
├── main.py
├── test.py
└── train.py
```

## Usage
### Requirements
* ``` python >= 3.5 ```
* ``` PyTorch >= 1.6.0 ``` <br>

For more detail of requirements: <br>
``` 
pip install -r requirements.txt 
```
### Data
The training data is split into five parts:
* ``` data/cifar-10-batches-py/data_batch_1 ```
* ``` data/cifar-10-batches-py/data_batch_2 ```
* ``` data/cifar-10-batches-py/data_batch_3 ```
* ``` data/cifar-10-batches-py/data_batch_4 ```
* ``` data/cifar-10-batches-py/data_batch_5 ``` <br>

The location of test data:
* ``` data/cifar-10-batches-py/test_batch ```
### Some important Parameters
```data_path``` 数据集的路径 <br>
```model_name``` 要训练的模型名称 <br>
```state_dir``` 要保存的模型的路径 <br>
```log_dir``` 要保存的日志的路径 <br>

```num_epoch``` 训练的总轮数，默认为200 <br>
```batch_size```指定训练阶段和测试阶段的batch_size,默认为64 <br>
```lr``` 学习率，默认为0.01 <br>
```optimizer``` 选择的优化器，如sgd,adam，默认为sgd <br>
```momentum``` SGD动量，默认为0.9 <br>

```gpu``` 训练机器的选择 <br>

## Running
### Train
```
python main.py --gpu cuda:0 \
    --num_epoch 200 \
    --batch_size 64 \
    --lr 0.001 \
    --optimizer sgd \
    --momentum 0.9 \
    --data_path ./data/ \
    --model_name my_model \
    --state_dir ./state \
    --log_dir ./log
```
### tensorboard
```
tensorboard --logdir runs
```
### Test
```
python test.py --gpu cuda:0 \
    --batch_size 64 \
    --model_name my_model \
    --data_path ./data/ \
    --state_dir ./state
```

## Results
### The results of various models on this classification task
model list:
* My own net : a cnn classification model designed by myself
* VGG : based on imagenet pretrained vgg16,vgg19
* Resnet : based on imagenet pretrained resnet18,resnet34,resnet50,resnet101 and so on
* Mobilenet : mobilenet and mobilenetv2
* Googlenet : A classic image network
* Densenet : Dense block +transition layer

| Model Name | Accuracy |
| ---------- | -------- |
|  my_model  |    85%   |
### Detailed results and analysis on my own model
First, here are the details of the parameters:
* Batch_size = 64
* epoch = 200
* Learning_rate: 1e-2, 1e-3
* optimizer: SGD, Adam <br>

Then,Below is a diagram of the training process: <br>
* On the left is the train_ACC diagram, the horizontal coordinate is epoch, and the vertical coordinate is accuracy; <br>
* On the right is the Loss diagram, the horizontal coordinate is EPOCH and the vertical coordinate is Loss. <br>

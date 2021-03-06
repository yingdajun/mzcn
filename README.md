
# mzcn

中文版本的matchzoo-py

本库包是基于matchzoo-py的库包做的二次开发开源项目，MatchZoo 是一个通用的文本匹配工具包，它旨在方便大家快速的实现、比较、以及分享最新的深度文本匹配模型。
<br>
由于matchzoo-py面向英文预处理较为容易，中文处理则需要进行一定的预处理。为此本人在借鉴学习他人成功的基础上，改进了matchzoo-py包，开发mzcn库包。
<br>
mzcn库包对中文文本语料进行只保留文本、去除表情、去除空格、去除停用词等操作，使得使用者可以快速进行中文文本语料进行预处理，使用方法和matchzoo-py基本一致。

# 快速入手

## 定义损失函数和指标


```python
import torch
import numpy as np
import pandas as pd
import mzcn as mz
print('matchzoo version', mz.__version__)
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)
```

    C:\Users\Administrator\Anaconda3\lib\requests\__init__.py:80: RequestsDependencyWarning: urllib3 (1.25.11) or chardet (3.0.4) doesn't match a supported version!
      RequestsDependencyWarning)
    

    matchzoo version 1.0.1
    `ranking_task` initialized with metrics [normalized_discounted_cumulative_gain@3(0.0), normalized_discounted_cumulative_gain@5(0.0), mean_average_precision(0.0)]
    

## 准备输入数据


```python
def load_data(tmp_data,tmp_task):
	df_data = mz.pack(tmp_data,task=tmp_task)
	return df_data
##数据集，并且进行相应的预处理
train=pd.read_csv('./data/train_data.csv').sample(100)
dev=pd.read_csv('./data/dev_data.csv').sample(50)
test=pd.read_csv('./data/test_data.csv').sample(30)
train_pack_raw = load_data(train,ranking_task)
dev_pack_raw = load_data(dev,ranking_task)
test_pack_raw=load_data(test,ranking_task)
```

# 停用表配置


```python
import os
```


```python
folder=mz.__path__[0]+'\\preprocessors\\units\\'
file=folder+'stopwords.txt'
if not os.path.exists(file):
    print('请将stopwords.txt放置在'+folder+'下面')
else:
    print('停用表配置成功')
```

    停用表配置成功
    

## 数据集预处理


```python
preprocessor = mz.models.aNMM.get_default_preprocessor()
```


```python
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)
```

    Processing text_left with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval:   0%| | 0/92 [00:00<?, ?it/s]Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\ADMINI~1\AppData\Local\Temp\jieba.cache
    Loading model cost 1.260 seconds.
    Prefix dict has been built succesfully.
    Processing text_left with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 92/92 [00:01<00:00, 51.43it/s]
    Processing text_right with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 94/94 [00:00<00:00, 183.34it/s]
    Processing text_right with append: 100%|████████████████████████████████████████████| 94/94 [00:00<00:00, 47020.22it/s]
    Building FrequencyFilter from a datapack.: 100%|████████████████████████████████████| 94/94 [00:00<00:00, 23511.51it/s]
    Processing text_right with transform: 100%|█████████████████████████████████████████| 94/94 [00:00<00:00, 23515.72it/s]
    Processing text_left with extend: 100%|█████████████████████████████████████████████| 92/92 [00:00<00:00, 92182.51it/s]
    Processing text_right with extend: 100%|████████████████████████████████████████████| 94/94 [00:00<00:00, 23529.76it/s]
    Building Vocabulary from a datapack.: 100%|████████████████████████████████████| 1003/1003 [00:00<00:00, 501955.25it/s]
    Processing text_left with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 92/92 [00:00<00:00, 185.96it/s]
    Processing text_right with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 94/94 [00:00<00:00, 187.73it/s]
    Processing text_right with transform: 100%|█████████████████████████████████████████| 94/94 [00:00<00:00, 94119.02it/s]
    Processing text_left with transform: 100%|██████████████████████████████████████████| 92/92 [00:00<00:00, 30661.58it/s]
    Processing text_right with transform: 100%|█████████████████████████████████████████| 94/94 [00:00<00:00, 23519.93it/s]
    Processing length_left with len: 100%|██████████████████████████████████████████████| 92/92 [00:00<00:00, 23001.67it/s]
    Processing length_right with len: 100%|█████████████████████████████████████████████| 94/94 [00:00<00:00, 93939.62it/s]
    Processing text_left with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 44/44 [00:00<00:00, 174.71it/s]
    Processing text_right with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 49/49 [00:00<00:00, 195.33it/s]
    Processing text_right with transform: 100%|█████████████████████████████████████████| 49/49 [00:00<00:00, 24440.59it/s]
    Processing text_left with transform: 100%|██████████████████████████████████████████| 44/44 [00:00<00:00, 44097.82it/s]
    Processing text_right with transform: 100%|█████████████████████████████████████████| 49/49 [00:00<00:00, 24423.16it/s]
    Processing length_left with len: 100%|██████████████████████████████████████████████| 44/44 [00:00<00:00, 14678.23it/s]
    Processing length_right with len: 100%|█████████████████████████████████████████████| 49/49 [00:00<00:00, 48910.26it/s]
    Processing text_left with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 30/30 [00:00<00:00, 166.77it/s]
    Processing text_right with chain_transform of ChineseRemoveBlack => ChineseSimplified => ChineseEmotion => IsChinese => ChineseStopRemoval => ChineseTokenizeDemo => Tokenize => Lowercase => PuncRemoval: 100%|█| 30/30 [00:00<00:00, 184.16it/s]
    Processing text_right with transform: 100%|█████████████████████████████████████████| 30/30 [00:00<00:00, 30124.28it/s]
    Processing text_left with transform: 100%|███████████████████████████████████████████| 30/30 [00:00<00:00, 5000.16it/s]
    Processing text_right with transform: 100%|█████████████████████████████████████████| 30/30 [00:00<00:00, 30030.82it/s]
    Processing length_left with len: 100%|███████████████████████████████████████████████| 30/30 [00:00<00:00, 9986.44it/s]
    Processing length_right with len: 100%|██████████████████████████████████████████████| 30/30 [00:00<00:00, 7513.98it/s]
    

## 生成训练数据


```python
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1
)
devset = mz.dataloader.Dataset(
    data_pack=dev_pack_processed
)
```

## 生成管道


```python
padding_callback = mz.models.aNMM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback,
)
devloader = mz.dataloader.DataLoader(
    dataset=devset,
    stage='dev',
    callback=padding_callback,
)
```

## 定义模型


```python
model = mz.models.aNMM()
model.params['task'] = ranking_task
model.params["embedding_output_dim"]=100
model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]
model.params['dropout_rate'] = 0.1
model.build()
print(model)
```

    aNMM(
      (embedding): Embedding(348, 100, padding_idx=0)
      (matching): Matching()
      (hidden_layers): Sequential(
        (0): Sequential(
          (0): Linear(in_features=200, out_features=100, bias=True)
          (1): ReLU()
        )
        (1): Sequential(
          (0): Linear(in_features=100, out_features=1, bias=True)
          (1): ReLU()
        )
      )
      (q_attention): Attention(
        (linear): Linear(in_features=100, out_features=1, bias=False)
      )
      (dropout): Dropout(p=0.1, inplace=False)
      (out): Linear(in_features=1, out_features=1, bias=True)
    )
    

## 模型训练


```python
optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=devloader,
    validate_interval=None,
    epochs=10
)

trainer.run()
```


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-1 Loss-1.003]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4154 - normalized_discounted_cumulative_gain@5(0.0): 0.4154 - mean_average_precision(0.0): 0.4107
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-2 Loss-1.003]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4154 - normalized_discounted_cumulative_gain@5(0.0): 0.4154 - mean_average_precision(0.0): 0.4107
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-3 Loss-1.001]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4154 - normalized_discounted_cumulative_gain@5(0.0): 0.4154 - mean_average_precision(0.0): 0.4107
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-4 Loss-1.002]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4154 - normalized_discounted_cumulative_gain@5(0.0): 0.4154 - mean_average_precision(0.0): 0.4107
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-5 Loss-1.001]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4286 - normalized_discounted_cumulative_gain@5(0.0): 0.4286 - mean_average_precision(0.0): 0.4286
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-6 Loss-1.001]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4286 - normalized_discounted_cumulative_gain@5(0.0): 0.4286 - mean_average_precision(0.0): 0.4286
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-7 Loss-0.998]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4286 - normalized_discounted_cumulative_gain@5(0.0): 0.4286 - mean_average_precision(0.0): 0.4286
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-8 Loss-1.000]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4286 - normalized_discounted_cumulative_gain@5(0.0): 0.4286 - mean_average_precision(0.0): 0.4286
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-9 Loss-0.999]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4286 - normalized_discounted_cumulative_gain@5(0.0): 0.4286 - mean_average_precision(0.0): 0.4286
    
    


    HBox(children=(IntProgress(value=0, max=1), HTML(value='')))


    [Iter-10 Loss-0.999]:
      Validation: normalized_discounted_cumulative_gain@3(0.0): 0.4286 - normalized_discounted_cumulative_gain@5(0.0): 0.4286 - mean_average_precision(0.0): 0.4286
    
    Cost time: 3.331088066101074s
    

# Install

由于mzcn是依赖于matchzoo-py模型，所以一共有两种途径安装mzcn

### Install MatchZoo-py from Pypi:
pip install mzcn

### Install MatchZoo-py from the Github source:

git clone https://github.com/yingdajun/mzcn.git
<br>
cd mzcn
<br>
python setup.py install

# Citation

本人是第一次写库包，水平有限，希望能给大家带来使用的帮助，如果有不足的地方请多指教
这里是所有引用过的库包

## matchzoo-py

@inproceedings{Guo:2019:MLP:3331184.3331403,
 author = {Guo, Jiafeng and Fan, Yixing and Ji, Xiang and Cheng, Xueqi},
 title = {MatchZoo: A Learning, Practicing, and Developing System for Neural Text Matching},
 booktitle = {Proceedings of the 42Nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR'19},
 year = {2019},
 isbn = {978-1-4503-6172-9},
 location = {Paris, France},
 pages = {1297--1300},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3331184.3331403},
 doi = {10.1145/3331184.3331403},
 acmid = {3331403},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {matchzoo, neural network, text matching},
} 

## CSDN的作者：SK-Berry的博文

https://blog.csdn.net/sk_berry/article/details/104984599

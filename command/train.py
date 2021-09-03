
# coding: utf-8

# # 训练

# In[ ]:



# coding: utf-8

# # 训练

# In[1]:


# coding: utf-8
import logging
import argparse
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import torch
import mzcn as mz

import time
 
from mzcn.models import *
from mzcn.tasks import *
from mzcn.losses import *
from mzcn.metrics import *

from mzcn.preprocessors import BasicPreprocessor
# 将所有model的配置文件加载进来
from config import *
# 将所有辅助性文件加载进来,
#特别是loadutils文件中的load_data、load_dict、load_json、dump
from utils import *
from utils.loadutils import *
from utils.loadmodel import *
from utils.onnxutils import *

def main(opt):
    config=opt.config_file
#     config=opt
    #加载所需要的模型的config文件，例如anmm是'config/anmm.json'
    ## 加载配置文件
    js_load=load_json(config)
    ## 加载模型类型
    model_class=js_load["model"]["model_class"]
    #加载损失/指标/模型/优化器/任务各类函数
    loss_dict=load_dict('utils/loss_dict.pkl')
    metric_dict=load_dict('utils/metric_dict.pkl')
    model_class_dict=load_dict('utils/model_class_dict.pkl')
    optimizer_dict=load_dict('utils/optimizer_dict.pkl')
    task_dict=load_dict('utils/task_dict.pkl')
    # 设置任务：
    #这个就可以设为opt的参数了，这个rank就设置为JSON格式里面的结构
    #设置任务的类型，损失函数，指标
    task_type=js_load['task']['name']
    task_loss=[]
    for i in js_load['task']['loss']:
        task_loss.append(loss_dict[i]())
    task_metrics=[]
    for i in js_load['task']['metrics']:
        task_metrics.append(metric_dict[i]())
    task = task_dict[task_type]()
    # mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
    task.metrics = task_metrics
    task.losses=task_loss
    # # 数据集载入
    data_url=js_load["data_url"]
    print('data loading ...')
    ##数据集，并且进行相应的预处理
    train=pd.read_csv(data_url[0]).sample(100)
    dev=pd.read_csv(data_url[1]).sample(50)
    test=pd.read_csv(data_url[2]).sample(30)
    train_pack_raw = load_data(train,task)
    dev_pack_raw = load_data(dev,task)
    test_pack_raw=load_data(test,task)
    # train_pack_raw = load_data(train,cls_task)
    # dev_pack_raw = load_data(dev,cls_task)
    # test_pack_raw=load_data(test,cls_task)
    model_class=js_load["model"]["model_class"]
    #加载预处理器
    pre=js_load["preprocessor"]
    preprocessor=load_preprocessor(pre,model_class)
    train_pack_processed = preprocessor.fit_transform(train_pack_raw)
    dev_pack_processed = preprocessor.transform(dev_pack_raw)
    test_pack_processed = preprocessor.transform(test_pack_raw)
    preprocessor.save('./result/pre/'+model_class+'/'+preprocessor.DATA_FILENAME)
    #加载embedding_matrix
    embedding_matrix=load_embedding(model_class,preprocessor)
    dataset=js_load["Dataset"]
    
    device=js_load["device"]
    callback=js_load["callback"]
    
    
    def case1():                            # 第一种情况执行的函数
        trainset,devset=load_dssm_dataset(dataset,
                                         model_class,
                                         train_pack_processed,
                                         dev_pack_processed,
                                         embedding_matrix=embedding_matrix,
                                         preprocessor=preprocessor)
        trainloader,devloader=load_dssm_loader(callback,model_class,
                                                trainset,devset,
                                                device,
                                                embedding_matrix,preprocessor)
        return trainloader,devloader
    
    def case2():                            # 第二种情况执行的函数
        trainset,devset=load_cdssm_dataset(dataset,
                                         model_class,
                                         train_pack_processed,
                                         dev_pack_processed,
                                         embedding_matrix=embedding_matrix,
                                         preprocessor=preprocessor)
        trainloader,devloader=load_cdssm_loader(callback,model_class,
                                                trainset,devset,
                                                device,
                                                embedding_matrix,preprocessor)
        return trainloader,devloader
    
    def case3():                            # 第一种情况执行的函数
        trainset,devset=load_duet_dataset(dataset,
                                         model_class,
                                         train_pack_processed,
                                         dev_pack_processed,
                                         embedding_matrix=embedding_matrix,
                                         preprocessor=preprocessor)
        trainloader,devloader=load_duet_loader(callback,model_class,
                                                trainset,devset,
                                                device,
                                                embedding_matrix,preprocessor)
        return trainloader,devloader
    
    def case4():                            # 第一种情况执行的函数
        trainset,devset=load_drmm_dataset(dataset,
                                         model_class,
                                         train_pack_processed,
                                         dev_pack_processed,
                                         embedding_matrix=embedding_matrix,
                                         preprocessor=preprocessor)
        trainloader,devloader=load_drmm_loader(callback,model_class,
                                                trainset,devset,
                                                device,
                                                embedding_matrix,preprocessor)
        return trainloader,devloader

    def default():                            # 默认情况下执行的函数
        trainset,devset=load_dataset(dataset,
                                     model_class,
                                     train_pack_processed,
                                     dev_pack_processed,
                                     embedding_matrix=embedding_matrix,
                                     preprocessor=preprocessor)
        trainloader,devloader =load_loader(callback,model_class
                                       ,trainset,devset,
                                       device,
                 embedding_matrix,preprocessor)
        
        return trainloader,devloader

    switch = {'dssm': case1,                # 注意此处不要加括号
              'cdssm': case2,
              'duet': case3,
              'drmm': case4
              }

    choice = model_class                         # 获取选择
    trainloader,devloader=switch.get(choice, default)()            # 执行对应的函数，如果没有就执行默认的函数
    
    #加载模型
    model_para=js_load["model"]
    func_dict=load_func_dict()
    model=func_dict[model_class](model_class_dict,model_class,task,model_para,preprocessor)
    print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optiz=js_load['optimizer']
    lr=js_load['lr']
    epoch=js_load['epoch']
    load_trainer(optimizer_dict,optiz,model,lr,epoch,trainloader,devloader)
    
    t=time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 
    export_model_file="./result/model_para/"+model_class+"_"+optiz+"_"+lr+"_"+epoch+"_"+t+"_model_parameter.pkl"
    torch.save(model.state_dict(), export_model_file)
    
    for batch in devloader:
        batch
        break
        
        
    bug_onnx_model=['dssm','cdssm','match_pyramid','duet','drmm']
    if model_class in bug_onnx_model:
        print('当前版本不支持'+model_class+'模型导出')
        else:
            if len(opt.onnx_file)!=0:
                onnx_file='result/'+model_class+"_"+t+'.onnx'
                exportOnnx(model,batch,onnx_file)
            print('随意从训练集中挑选一个batch的数据集的预测结果如下')
            pred=loadOnnx(batch,onnx_file)
            print(pred)
    
if __name__ == '__main__':
    folder=mz.__path__[0]+'\\preprocessors\\units\\'
    file=folder+'stopwords.txt'
    if not os.path.exists(file):
        print('请将stopwords.txt放置在'+folder+'下面'+'否则会报错')
    else:
        print('停用表配置成功')
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    #设置训练参数
    parser.add_argument('--config_file', default='config/anmm.json', type=str)
    parser.add_argument('--onnx_file', default='result/anmm.onnx', type=str)
    opt = parser.parse_args()
    main(opt)




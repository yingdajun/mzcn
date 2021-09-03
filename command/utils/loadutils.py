#!/usr/bin/env python
# coding: utf-8
import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import torch
import mzcn as mz

# 这些属于辅助的文件，统一放置在这里
from mzcn.models import *
from mzcn.tasks import *
from mzcn.losses import *
from mzcn.metrics import *
# 加载处理器
from mzcn.preprocessors import BasicPreprocessor
# from mzcn.preprocessors import BasicPreprocessor
from mzcn.dataloader.callbacks import BasicPadding,DRMMPadding

#将得到的pandas文件转化成mzcn所需要的格式
def load_data(tmp_data,tmp_task):
	df_data = mz.pack(tmp_data,task=tmp_task)
	return df_data

#加载指标等函数的PKL文件
def load_dict(file):
    fp=open(file,'rb')
    tmp_dict=pickle.load(fp,encoding='gbk')
    print(tmp_dict)
    print('='*50)
    return tmp_dict

# 加载JSON模型
def load_json(file):
    # 加载JSON格式
    # 读取json文件内容,返回字典格式
    with open(file,'r')as fp:
        t1 = json.load(fp)
#     t=json.loads(t1)
#     t1=json.dumps(dic, indent=4, separators=(',', ':')) 
    return t1

# 保存JSON模型
def dump(t,file):
    js = json.dumps(t, indent=4, separators=(',', ':')) 
    fileObject = open(file, 'w', encoding='utf-8')  
    fileObject.write(js)  
    fileObject.close() 



def load_cdssm_dataset(dataset,model_class,train_pack_processed,dev_pack_processed,
                 embedding_matrix,preprocessor):
    triletter_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='sum')
    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        mode=dataset["mode"],
        num_dup=dataset["num_dup"],
        num_neg=dataset["num_neg"],
        batch_size=dataset["batch_size"],
        shuffle=dataset["shuffle"],
        sort=dataset["sort"],
        callbacks=[triletter_callback]
    )
    devset = mz.dataloader.Dataset(
        data_pack=dev_pack_processed,
        callbacks=[triletter_callback]
    )
    return trainset,devset 


def load_embedding(model_class,preprocessor):
    glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=50)
    term_index = preprocessor.context['vocab_unit'].state['term_index']
    embedding_matrix = glove_embedding.build_matrix(term_index)
    l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
    embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]
    return embedding_matrix


def load_dataset(dataset,model_class,train_pack_processed,dev_pack_processed,
                 embedding_matrix,preprocessor):
    if model_class== 'drmm':
        embedding_matrix=load_embedding(model_class,preprocessor)
        histgram_callback = mz.dataloader.callbacks.Histogram(
            embedding_matrix, bin_size=30, hist_mode='LCH'
        )
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode=dataset["mode"],
            num_dup=dataset["num_dup"],
            num_neg=dataset["num_neg"],
            batch_size=dataset["batch_size"],
            shuffle=dataset["shuffle"],
            sort=dataset["sort"],
            callbacks=[histgram_callback]
        )
        devset = mz.dataloader.Dataset(
            data_pack=dev_pack_processed,
            callbacks=[histgram_callback]
        )
    trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode=dataset["mode"],
    num_dup=dataset["num_dup"],
    num_neg=dataset["num_neg"],
    batch_size=dataset["batch_size"],
    shuffle=dataset["shuffle"],
    sort=dataset["sort"],
    callbacks=dataset["callbacks"]
    )
    devset = mz.dataloader.Dataset(
        data_pack=dev_pack_processed
    )       
    return trainset,devset 

def load_cdssm_dataset(dataset,model_class,train_pack_processed,dev_pack_processed,
                 embedding_matrix,preprocessor):
    triletter_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='sum')
#         dataset=js_load["Dataset"]
    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        mode=dataset["mode"],
        num_dup=dataset["num_dup"],
        num_neg=dataset["num_neg"],
        batch_size=dataset["batch_size"],
        shuffle=dataset["shuffle"],
        sort=dataset["sort"],
        callbacks=[triletter_callback]
    )
    devset = mz.dataloader.Dataset(
        data_pack=dev_pack_processed,
        callbacks=[triletter_callback]
    )   
    return trainset,devset 

def load_dssm_dataset(dataset,model_class,train_pack_processed,dev_pack_processed,
                 embedding_matrix,preprocessor):
    triletter_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='aggregate')
    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        mode=dataset["mode"],
        num_dup=dataset["num_dup"],
        num_neg=dataset["num_neg"],
        batch_size=dataset["batch_size"],
        shuffle=dataset["shuffle"],
        sort=dataset["sort"],
        callbacks=[triletter_callback]
    )
    devset = mz.dataloader.Dataset(
        data_pack=dev_pack_processed,
        callbacks=[triletter_callback]
    )
    return trainset,devset 

def load_duet_dataset(dataset,model_class,train_pack_processed,dev_pack_processed,
                 embedding_matrix,preprocessor):
    triletter_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='sum')
#         dataset=js_load["Dataset"]
    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        mode=dataset["mode"],
        num_dup=dataset["num_dup"],
        num_neg=dataset["num_neg"],
        batch_size=dataset["batch_size"],
        shuffle=dataset["shuffle"],
        sort=dataset["sort"],
        callbacks=[triletter_callback]
    )
    devset = mz.dataloader.Dataset(
        data_pack=dev_pack_processed,
        callbacks=[triletter_callback]
    )   
    return trainset,devset 

def load_drmm_dataset(dataset,model_class,train_pack_processed,dev_pack_processed,
                 embedding_matrix,preprocessor):
    histgram_callback = mz.dataloader.callbacks.Histogram(
    embedding_matrix, bin_size=30, hist_mode='LCH'
    )
    trainset = mz.dataloader.Dataset(
        data_pack=train_pack_processed,
        mode=dataset["mode"],
        num_dup=dataset["num_dup"],
        num_neg=dataset["num_neg"],
        batch_size=dataset["batch_size"],
        shuffle=dataset["shuffle"],
        sort=dataset["sort"],
        callbacks=[histgram_callback]
    )
    devset = mz.dataloader.Dataset(
        data_pack=dev_pack_processed,
        callbacks=[histgram_callback]
    )
    return trainset,devset 

# if model_class== 'cdssm' or model_class=='dssm'or model_class=='duet':


def load_loader(callback,model_class,trainset,devset,device,
                 embedding_matrix,preprocessor):

    if model_class== 'drmm':
        padding_callback = DRMMPadding(
        fixed_length_left=callback["fixed_length_left"],
        fixed_length_right=callback["fixed_length_right"],
        )
       
    padding_callback = BasicPadding(
    fixed_length_left=callback["fixed_length_left"],
    fixed_length_right=callback["fixed_length_right"],
    pad_word_value=callback["pad_word_value"],
    pad_word_mode=callback["pad_word_mode"],
    fixed_ngram_length=callback["fixed_ngram_length"]
    )

    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        stage='train',
        callback=padding_callback,
        device=device
    )

    devloader = mz.dataloader.DataLoader(
        dataset=devset,
        stage='dev',
        callback=padding_callback,
        device=device
    )
    return trainloader,devloader


def load_dssm_loader(callback,model_class,trainset,devset,device,
                 embedding_matrix,preprocessor):
    padding_callback = mz.models.DSSM.get_default_padding_callback(
    #     fixed_length_left=10,
    # #     callback["fixed_length_left"],
    #     fixed_length_right=100,
    #     callback["fixed_length_right"],
    #     pad_word_value=callback["pad_word_value"],
    #     pad_word_mode=callback["pad_word_mode"],
    #     fixed_ngram_length=callback["fixed_ngram_length"]
    )
    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        stage='train',
        callback=padding_callback,
        device=device
    )
    devloader = mz.dataloader.DataLoader(
        dataset=devset,
        stage='dev',
        callback=padding_callback,
        device=device
    )
    return trainloader,devloader

def load_cdssm_loader(callback,model_class,trainset,devset,device,
                 embedding_matrix,preprocessor):
    padding_callback = mz.models.CDSSM.get_default_padding_callback(
            fixed_length_left=callback["fixed_length_left"],
            fixed_length_right=callback["fixed_length_right"],
            pad_word_value=callback["pad_word_value"],
            pad_word_mode=callback["pad_word_mode"],
            fixed_ngram_length=preprocessor.context['ngram_vocab_size']
        )
    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        stage='train',
        callback=padding_callback,
        device=device
    )

    devloader = mz.dataloader.DataLoader(
        dataset=devset,
        stage='dev',
        callback=padding_callback,
        device=device
    )
    return trainloader,devloader


def load_duet_loader(callback,model_class,trainset,devset,device,
                 embedding_matrix,preprocessor):
    padding_callback = mz.models.DUET.get_default_padding_callback(
    fixed_length_left=callback["fixed_length_left"],
    fixed_length_right=callback["fixed_length_right"],
    pad_word_value=callback["pad_word_value"],
    pad_word_mode=callback["pad_word_mode"],
    fixed_ngram_length=callback["fixed_ngram_length"],
    with_ngram=callback["with_ngram"]
    )
    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        stage='train',
        callback=padding_callback,
        device=device
    )
    devloader = mz.dataloader.DataLoader(
        dataset=devset,
        stage='dev',
        callback=padding_callback,
        device=device
    )
    return trainloader,devloader

def load_drmm_loader(callback,model_class,trainset,devset,device,
                 embedding_matrix,preprocessor):
    padding_callback = mz.models.DRMM.get_default_padding_callback(
    fixed_length_left=callback["fixed_length_left"],
    fixed_length_right=callback["fixed_length_right"],
    #     pad_word_value=callback["pad_word_value"],
    #     pad_word_mode=callback["pad_word_mode"],
    #     fixed_ngram_length=callback["fixed_ngram_length"]
    )
    trainloader = mz.dataloader.DataLoader(
        dataset=trainset,
        stage='train',
        callback=padding_callback,
        device=device
    )
    devloader = mz.dataloader.DataLoader(
        dataset=devset,
        stage='dev',
        callback=padding_callback,
        device=device
    )
    return trainloader,devloader

# 加载处理器
def load_preprocessor(pre,model_class):
    preprocessor =BasicPreprocessor(
         truncated_length_left=pre["truncated_length_left"],
         truncated_length_right=pre["truncated_length_right"],
         filter_mode=pre["filter_mode"],
         filter_low_freq=pre["filter_low_freq"],
         filter_high_freq=pre["filter_high_freq"],
         remove_stop_words=pre["remove_stop_words"],
         ngram_size=pre['ngram_size']
        )
    return preprocessor
    
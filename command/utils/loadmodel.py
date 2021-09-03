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
# from mzcn.preprocessors import BasicPreprocessor
from mzcn.dataloader.callbacks import BasicPadding,DRMMPadding



def load_func_dict():
    func_key=['anmm','arci', 'arcii','cdssm','conv_knrm',
     'drmm', 'drmmtks','dssm','duet','esim', 'knrm',
     'match_pyramid','match_srnn', 'matchlstm' 
    ]

    func_value=[load_anmm,load_arci,load_arcii,load_cdssm,load_conv_knrm,
     load_drmm,load_drmmtks,load_dssm,load_duet,load_esim,load_knrm,
    load_match_pyramid,load_match_srnn,load_matchlstm]

    func_dict=dict(zip(func_key,func_value))
    return func_dict


def load_embedding(model_class,preprocessor):
    embedding_matrix=None
    if model_class=='drmm':
        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=50)
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = glove_embedding.build_matrix(term_index)
        l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
        embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]  
    return embedding_matrix

def load_anmm(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]
    # 不是所有模型都需要下面的参数的
    # model.params['left_length'] = 20
    # model.params['right_length'] = 100
    model.build()
    print(model)
    return model

def load_arci(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]

    model.params['left_filters'] = model_para["left_filters"]
    model.params['right_filters'] = model_para["right_filters"]
    model.params['left_kernel_sizes'] = model_para["left_kernel_sizes"]
    model.params['right_kernel_sizes'] = model_para["right_kernel_sizes"]
    model.params['left_pool_sizes'] = model_para["left_pool_sizes"]
    model.params['right_pool_sizes'] = model_para["right_pool_sizes"]
    model.params['conv_activation_func'] = model_para["conv_activation_func"]
    model.params['mlp_num_layers'] = model_para["mlp_num_layers"]
    model.params['mlp_num_units'] = model_para["mlp_num_units"]
    model.params['mlp_num_fan_out'] = model_para["mlp_num_fan_out"]
    model.params['mlp_activation_func'] = model_para["mlp_activation_func"]
    model.params['dropout_rate'] = model_para["dropout_rate"]
    # 不是所有模型都需要下面的参数的
    # model.params['left_length'] = 20
    # model.params['right_length'] = 100

    model.build()
    print(model)
    return model

def load_arcii(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]
    model.params['left_length'] = model_para['left_length']
    model.params['right_length'] = model_para['right_length']
    model.params['kernel_1d_count'] = model_para["kernel_1d_count"]
    model.params['kernel_1d_size'] = model_para["kernel_1d_size"]
    model.params['kernel_2d_count'] = model_para["kernel_2d_count"]
    model.params['kernel_2d_size'] = model_para["kernel_2d_size"]
    model.params['pool_2d_size'] = model_para["pool_2d_size"]

    # 不是所有模型都需要下面的参数的
    # model.params['left_length'] = 20
    # model.params['right_length'] = 100

    model.build()
    print(model)
    return model

def load_cdssm(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    model.params['vocab_size'] = preprocessor.context['ngram_vocab_size']
    model.params['filters'] = model_para['filters']
    model.params['kernel_size'] = model_para['kernel_size']
    model.params['conv_activation_func'] = model_para['conv_activation_func']
    model.params['mlp_num_layers'] = model_para['mlp_num_layers']
    model.params['mlp_num_units'] = model_para['mlp_num_units']
    model.params['mlp_num_fan_out'] = model_para['mlp_num_fan_out']
    model.params['mlp_activation_func'] = model_para['mlp_activation_func']
    model.params['dropout_rate'] = model_para['dropout_rate']

    model.build()
    print(model)
    return model

def load_conv_knrm(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]
    model.params['filters'] = model_para['filters']
    model.params['conv_activation_func'] = model_para['conv_activation_func']
    model.params['max_ngram'] = model_para['max_ngram']
    model.params['use_crossmatch'] = model_para['use_crossmatch']
    model.params['kernel_num']  = model_para['kernel_num'] 
    model.params['sigma'] = model_para['sigma'] 
    model.params['exact_sigma']= model_para['exact_sigma']

    model.build()
    print(model)
    return model

def load_drmm(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]
    model.params['hist_bin_size'] = model_para['hist_bin_size']
    model.params['mlp_num_layers'] = model_para['mlp_num_layers']
    model.params['mlp_num_units'] = model_para['mlp_num_units']
    model.params['mlp_num_fan_out'] = model_para['mlp_num_fan_out']
    model.params['mlp_activation_func'] = model_para['mlp_activation_func']

    model.build()
    print(model)
    return model

def load_drmmtks(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]

    model.params['top_k'] = model_para['top_k']
    model.params['mask_value']= model_para['mask_value']
    model.params['mlp_num_layers'] = model_para["mlp_num_layers"]
    model.params['mlp_num_units'] = model_para["mlp_num_units"]
    model.params['mlp_num_fan_out'] = model_para["mlp_num_fan_out"]
    model.params['mlp_activation_func'] = model_para["mlp_activation_func"]

    model.build()
    print(model)
    return model

def load_dssm(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params['vocab_size']=preprocessor.context['vocab_size']

    model.params['mlp_num_layers'] = model_para['mlp_num_layers']
    model.params['mlp_num_units'] = model_para['mlp_num_units']
    model.params['mlp_num_fan_out']  = model_para['mlp_num_fan_out'] 
    model.params['mlp_activation_func'] = model_para['mlp_activation_func'] 

    model.build()
    model=model.double()
    print(model)
    return model

def load_duet(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params['vocab_size'] = preprocessor.context['ngram_vocab_size']

    model.params['left_length'] = model_para['left_length']
    model.params['right_length'] = model_para['right_length']
    model.params['lm_filters']= model_para['lm_filters']
    model.params['mlp_num_layers']= model_para['mlp_num_layers']
    model.params['mlp_num_units']= model_para['mlp_num_units']
    model.params['mlp_num_fan_out']= model_para['mlp_num_fan_out']
    model.params['mlp_activation_func'] = model_para['mlp_activation_func'] 
    model.params['dm_conv_activation_func'] = model_para['dm_conv_activation_func'] 
    model.params['dm_filters']= model_para['dm_filters']
    model.params['dm_kernel_size'] = model_para['dm_kernel_size'] 
    model.params['dm_right_pool_size'] = model_para['dm_right_pool_size'] 
    model.params['dropout_rate'] = model_para["dropout_rate"]

    model.build()
    print(model)
    return model

def load_esim(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]

    model.params['mask_value'] = model_para['mask_value']
    model.params['hidden_size']= model_para['hidden_size']
    model.params['lstm_layer'] = model_para['lstm_layer']
    model.params['dropout'] = model_para['dropout']

    model.build()
    print(model)
    return model

def load_knrm(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]

    model.params['kernel_num']= model_para['kernel_num']

    model.params['sigma'] = model_para['sigma'] 

    model.params['exact_sigma'] = model_para['exact_sigma'] 

    model.build()
    print(model)
    return model

def load_match_pyramid(model_class_dict,model_class,task,model_para,preprocessor):
    
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]

    model.params['kernel_count']  = model_para['kernel_count'] 

    model.params['kernel_size'] = model_para['kernel_size'] 

    model.params['dpool_size'] = model_para['dpool_size'] 

    model.params['dropout_rate'] = model_para['dropout_rate'] 

    model.build()
    print(model)
    return model

def load_match_srnn(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]

    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]

    model.params['channels']= model_para['channels']
    model.params['units'] = model_para['units'] 

    model.params['dropout'] = model_para['dropout'] 

    model.params['direction']= model_para['direction']

    model.build()
    print(model)
    return model

def load_matchlstm(model_class_dict,model_class,task,model_para,preprocessor):
    model = model_class_dict[model_class]()
    model.params['task'] = task
    # model.params['embedding'] = embedding_matrix #这里是当加载glove等模型时取消该行注释
    #设置embedding系数
    # 有些函数根本就没有这个
    model.params["embedding_output_dim"]=model_para["embedding_output_dim"]
    model.params["embedding_input_dim"]=preprocessor.context["embedding_input_dim"]
    model.params['dropout']= model_para['dropout']
    model.params['hidden_size'] = model_para['hidden_size'] 
    model.params['mask_value']= model_para['mask_value']

    model.build()
    print(model)
    return model


def load_trainer(optimizer_dict,optiz,model,lr,epoch,trainloader,devloader):
    optimizer = optimizer_dict[optiz](model.parameters(),lr=lr)
    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=trainloader,
        validloader=devloader,
        validate_interval=None,
        epochs=epoch
    )
    trainer.run()
#!/usr/bin/env python
# coding: utf-8
import numpy
import torch
import onnx
import onnxruntime as rt

#导入ONNX
def loadOnnx(inputbatch,file):
    # Load the ONNX model
    model = onnx.load(file)
    # Check that the IR is well formed
#     print(onnx.checker.check_model(model))
    onnx.helper.printable_graph(model.graph)
    # Compute the prediction with ONNX Runtime
    sess = rt.InferenceSession(file)
    input_name = sess.get_inputs()[0].name
    input_name1 = sess.get_inputs()[1].name
    label_name = sess.get_outputs()[0].name
    
    print('='*50)
    d=inputbatch
    di1=d['text_left']
    di2=d['text_right']
    pred_onx = sess.run([label_name], {input_name:di1.numpy()
                                       ,input_name1:di2.numpy()})[0]
    return pred_onx

# 导出ONNX
def exportOnnx(exportmodel,inputbatch,file):
    # Export the model
    torch.onnx.export(exportmodel,               # model being run
                      inputbatch,                         # model input (or a tuple for multiple inputs)
                      file,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
    #                   dynamic_axes=dynamic_axes
    #                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
    #                                 'output' : {0 : 'batch_size'}
    #                                }
                     )
    print('导出成功')



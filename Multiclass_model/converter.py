import torch
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import warnings

import sys
warnings.filterwarnings("ignore")


from tinynn.converter import TFLiteConverter
from tinynn.converter.base import GraphOptimizer
from config.config_loader import load_config
from model import BaselineCNN

def converter(
    model: torch.nn.modules,
    output_path: str,
    dummy_input: torch.Tensor
):
    # When converting quantized models, please ensure the quantization backend is set.
    torch.backends.quantized.engine = 'qnnpack'

    # The code section below is used to convert the model to the TFLite format
    # When `preserve_tensors=True` is specified, the intermediate tensors will be preserved,
    # so that they can be compared with those generated with other backends.
    # You may also need to tune the optimize level to adjust the granularity of the comparison.
    # For example, using values like `GraphOptimizer.FOLD_BUFFER` or `GraphOptimizer.NO_OPTIMIZE`
    # will ensure comparsion of the outputs in almost every layer,
    # while with `GraphOptimizer.ALL_OPTIMIZE` or `GraphOptimizer.COMMON_OPTIMIZE`,
    # some intermediate layers will be skipped because they may be fused with other layers.
    converter = TFLiteConverter(
        model, dummy_input, output_path,
        float16_quantization= True, 
        preserve_tensors=True, optimize=GraphOptimizer.ALL_OPTIMIZE,
    )
    converter.convert()

def main():
    config = load_config()
    
    MODEL_DIR = config['DIRECTORY']['BEST_MODEL_PATH']
    checkpoint  = torch.load(MODEL_DIR)
    
    model = BaselineCNN(config['BACKBONE_OPTION'])
    model.load_state_dict(checkpoint["model"])
    model = model.to(config['PARAMETER']['DEVICE']) 
    model.eval()   
    
    output_path = os.path.join(config['CONVERTER_OPTION']['CONVERT_MODEL_PATH'], config['CONVERTER_OPTION']['SAVE_FOLDER'], config['CONVERTER_OPTION']['TFLITE_MODEL'])
    dummy_input = torch.rand((1, 3, config['CONVERTER_OPTION']['DUMMY_HEIGHT'], config['CONVERTER_OPTION']['DUMMY_WIDTH']))
        
    converter(model, output_path, dummy_input)
        
    
#-----------------------------------------------------
if __name__ == "__main__":
    main()
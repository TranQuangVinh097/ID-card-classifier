

import torch

import pandas as pd
import os, shutil
from data.dataset import MyDataset
from model import BaselineCNN
from torch.utils.data import DataLoader 
import copy
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from config.config_loader import load_config
from matplotlib import pyplot as plt
from tools.engine import update_metric, get_metric
import cv2
import random
import time
from data.aug_option import aug_val

def tflite_model_inference(
    interpreter: tf.lite.Interpreter,
    tflite_input: np.array
):
    # Load TFLite model and allocate tensors.
    
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Test model on random input data.
    interpreter.set_tensor(input_details[0]['index'], tflite_input)
            
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = [[1 - output_data[0][i] if output_data[0][i] > output_data[0][i + 1] else output_data[0][i + 1] for i in range(0,6,2)]]

    return output_data


def compare_result(
    tfl_result: np.array,
    torch_result: np.array,
    atol=1e-5,
    rtol=1e-3,
    thres=0.9,
):
    """
    Compare the results of two arrays and print the comparison metrics.

    Args:
        tfl_result (np.array): The result from TensorFlow.
        torch_result (np.array): The result from PyTorch.
        atol (float, optional): The absolute tolerance for element-wise comparison. Defaults to 1e-5.
        rtol (float, optional): The relative tolerance for element-wise comparison. Defaults to 1e-3.
        thres (float, optional): The threshold value to convert results to binary. Defaults to 0.9.
    """
    # Convert results to binary using the threshold
    tfl_result[tfl_result > thres] = 1
    tfl_result[tfl_result <= thres] = 0
    torch_result[torch_result > thres] = 1
    torch_result[torch_result <= thres] = 0
    
    # Compare the results using np.allclose
    matches = np.allclose(tfl_result, torch_result, rtol=rtol, atol=atol)
    print(f'Output value matches: {matches}')

    # Calculate absolute difference
    diff = np.abs(torch_result - tfl_result)

    diff_mean = np.mean(diff)
    diff_min = np.min(diff)
    diff_max = np.max(diff)

    # Calculate the percentage of elements with absolute difference greater than the tolerance
    abs_err_percent = np.mean((diff > atol).astype('float32')) * 100
    print(
        f'Output absolute difference min,mean,max: {diff_min},{diff_mean},{diff_max} (error:'
        f' {abs_err_percent:.2f}%)'
    )

    # Calculate relative difference
    torch_result_nonzero = (torch_result != 0).astype('bool')
    if np.all(~torch_result_nonzero):
        rel_err = np.array([float('inf')] * len(torch_result))
    else:
        rel_err = diff[torch_result_nonzero] / np.abs(torch_result[torch_result_nonzero])

        rel_diff_mean = np.mean(rel_err)
        rel_diff_min = np.min(rel_err)
        rel_diff_max = np.max(rel_err)

        # Calculate the percentage of elements with relative difference greater than the tolerance
        rel_err_percent = np.mean((rel_err > rtol).astype('float32')) * 100
        print(
            f'Output relative difference min,mean,max: {rel_diff_min},{rel_diff_mean},{rel_diff_max} (error:'
            f' {rel_err_percent:.2f}%)'
        )

import copy

def test_model(f1_torch: list, f1_pred: list, f1_label: list, keyword: list):
    """
    Calculate and print the F-score, precision, and recall for each label,
    and then calculate and print the average F-score.

    Parameters:
        f1_torch (list): List of torch values.
        f1_pred (list): List of predicted values.
        f1_label (list): List of label values.
        keyword (list): List of keywords.

    Returns:
        None
    """
    
    thres_f_score = 0
    saved_thres = 0
    metric = 0
    f1_pred_cur = copy.deepcopy(f1_pred)
    
    for i in range(len(f1_label)):
        precision, recall, f_score = get_metric(i, f1_pred_cur, f1_label, 0.9)
        
        print(f'Result of {keyword[i]}: ')
        print(f'F-score: {f_score:.4f} Precision: {precision:.4f}, Recall: {recall:.4f} \n')
        metric += f_score
            
    metric /= len(f1_label)
    print(f'Average F-score: {metric:.4f}')
                

def tune_threshold(f1_torch, f1_pred, f1_label, multiply):
    """
    Tune the threshold for f1 score calculation.
    
    Args:
        f1_torch (list): List of torch f1 score values.
        f1_pred (list): List of predicted f1 score values.
        f1_label (list): List of f1 score labels.
        multiply (int): The multiplier for the threshold.
        
    Returns:
        None
    """
    end = np.power(10, multiply) - 1
    start = np.power(1, multiply - 1)
    multiply = np.power(10, multiply)
     
    thres_result = pd.DataFrame(columns=['Average f1 score', 'F1 score', 'Precision', 'Recall', 'Threshold'])
    for k in range(start, end, 1):
        metric = 0
        thres = k / multiply
        saved_metric = [[], [], []]
        f1_pred_cur = copy.deepcopy(f1_pred)
        
        # Calculate f1 score for each label
        for i in range(len(f1_label)):
            saved_metric = np.append(saved_metric,np.transpose(np.array([get_metric(i, f1_pred_cur, f1_label, thres)])),axis = 1)
            metric += saved_metric[2][len(saved_metric[2]) - 1]
            
        metric /= len(f1_label)
        
        new_row = {'Average f1 score': metric, 'F1 score': ['%.2f' % e for e in saved_metric[2]], 'Precision': ['%.2f' % e for e in saved_metric[0]], 'Recall': ['%.2f' % e for e in saved_metric[1]], 'Threshold': thres}
        thres_result.loc[len(thres_result)] = new_row
                
    thres_result.set_index('Threshold', inplace=True)
    print(thres_result)

def inference(
    model: torch.nn.Module,
    test_loader: DataLoader,
    data_dir: str,
    interpreter: tf.lite.Interpreter,
    config = None,
    mode = 'test',
):
    """
    Run inference on the given model using the validation loader.
    
    Args:
        model (torch.nn.Module): The model to perform inference with.
        test_loader (DataLoader): The validation loader containing the data to infer on.
        data_dir (str): The directory where the data is located.
        interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter for running inference.
        config (Optional): Additional configuration options. Default is None.
        mode (str): The mode of inference. Default is 'test'.
    """

    f1_pred = [[] for i in range(3)]  # Initialize lists for storing prediction results
    f1_torch = [[] for i in range(3)]  # Initialize lists for storing Torch results
    f1_label = [[] for i in range(3)]  # Initialize lists for storing label results
    total = 0  # Initialize total time counter
    h = config['CONVERTER_OPTION']['DUMMY_HEIGHT']  # Get height from config
    w = config['CONVERTER_OPTION']['DUMMY_WIDTH']  # Get width from config
    
    with torch.no_grad():  # Disable gradient calculation
        for inputs,labels in test_loader:  # Iterate over validation loader
            inputs = inputs.cuda().half()  # Move inputs to GPU and change data type to half precision
            labels = labels.cuda()  # Move labels to GPU
            outputs = model(inputs)  # Perform forward pass on model to get outputs
            
            # Calculate prediction probabilities from outputs
            pred = [
                [(1 - outputs[0][i]).item() if outputs[0][i] > outputs[0][i + 1] else (outputs[0][i + 1]).item() for i in range(0,6,2)]
            ]
            
            # Convert inputs to TensorFlow Lite input format
            tflite_input = inputs.type(torch.uint8).detach().cpu().numpy()
            tflite_input = np.transpose(tflite_input[0], (1,2,0))
            tflite_input = cv2.resize(tflite_input,(h, w))
            tflite_input = np.reshape(tflite_input,(1,h, w,3)).astype(np.float32) 
            
            t1 = time.time()  # Start timer
            output_data = tflite_model_inference(interpreter, tflite_input)  # Run TensorFlow Lite inference
            t2 = time.time()  # Stop timer
            
            total += t2 - t1  # Accumulate total time
            
            f1_pred = update_metric(output_data, f1_pred)  # Update prediction metric
            f1_torch = update_metric(pred, f1_torch)  # Update Torch metric
            f1_label = update_metric(labels, f1_label)  # Update label metric
   
    print('Total time: ', total / len(test_loader))  # Print average inference time
    compare_result(f1_pred, f1_torch)  # Compare prediction results
    
    if (mode == 'threshold'):  # Check if mode is 'threshold'
        tune_threshold(f1_pred, f1_torch, f1_label, 1)  # Perform threshold tuning
    else:
        test_model(f1_torch, f1_pred, f1_label, config['LABEL_OPTION']['LABEL'])  # Test model using metrics
    
    


 
def main():
    # Load configuration
    config = load_config()
    
    # Set file paths
    csv_path = config['DIRECTORY']['TEST_CSV']
    data_dir = config['DIRECTORY']['TEST_DATA_PATH']
    model_dir = config['DIRECTORY']['BEST_MODEL_PATH']
    
    # Print TFLite model path
    print(config['DIRECTORY']['TFLITE_MODEL_PATH'] + config['CONVERTER_OPTION']['TFLITE_MODEL'])
    
    # Load TFLite model interpreter
    interpreter = tf.lite.Interpreter(model_path=config['DIRECTORY']['TFLITE_MODEL_PATH'] + config['CONVERTER_OPTION']['TFLITE_MODEL'])
    
    # Read test CSV file
    test_df = pd.read_csv(csv_path)
    
    # Create test dataset
    test_dataset = MyDataset(data_dir, test_df, aug_val)
    
    # Create test data loader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
    )
    
    # Load trained model checkpoint
    checkpoint = torch.load(model_dir)
    
    # Create model instance
    model = BaselineCNN(config['BACKBONE_OPTION'])
    
    # Load trained model weights
    model.load_state_dict(checkpoint["model"])
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to specified device
    model.to(config['PARAMETER']['DEVICE'])
    
    # Perform inference
    inference(model, test_loader, data_dir, interpreter, config, mode='threshold')
    
if __name__ == "__main__":
    main()
    
    

# -*- coding: utf-8 -*-
import copy
import numpy as np


def reconstruct_by_iteration(x: np.ndarray,
                             fault_sensor_index: int,
                             pca_info: dict,
                             max_iteration: int = None):
    if max_iteration is None:
        max_iteration = 10000

    x_copy = copy.deepcopy(x)
    fault_sensor = x_copy[:, fault_sensor_index]
    
    reconstructed_sensor = predict_pca(pca_info=pca_info, data=x_copy)[:, fault_sensor_index]

    iteration = 0
    while abs(fault_sensor - reconstructed_sensor) > 0.0001:
        x_copy[:, fault_sensor_index] = reconstructed_sensor[0]    # reconstructed_sensor = [value]

        reconstructed_sensor = predict_pca(pca_info=pca_info, data=x_copy)[:, fault_sensor_index]

        iteration += 1
        if iteration > max_iteration:
            print('Reconstruction halted by Max iter')
            break

    reconstructed_x = predict_pca(pca_info=pca_info, data=x_copy)
    print(f"Reconstruct X: {x[:, fault_sensor_index]} ---> {reconstructed_x[:, fault_sensor_index]}")
    return reconstructed_x


def predict_pca(pca_info, data):
    x_mean = np.array(pca_info['x_mean'])
    x_std = np.array(pca_info['x_std'])
    loading_p = pca_info['loading_p']

    x_norm = (data - x_mean) / x_std
    
    t = np.dot(x_norm, loading_p.T)
    x_hat_norm = np.dot(t, loading_p)
    x_hat = x_hat_norm * x_std + x_mean
    
    return x_hat

# -*- coding: utf-8 -*-
import copy
import numpy as np


def reconstruct_by_iteration(x: np.ndarray,
                             fault_sensor_index: int,
                             pca_model,
                             max_iteration: int = None):
    if max_iteration is None:
        max_iteration = 10000

    x_copy = copy.deepcopy(x)
    fault_sensor = x[fault_sensor_index]
    reconstructed_sensor = pca_model.predict(x)[fault_sensor_index]

    iteration = 0
    while abs(fault_sensor - reconstructed_sensor) > 0.0001:
        x_copy[fault_sensor] = reconstructed_sensor

        reconstructed_sensor = pca_model.predict(x_copy)[fault_sensor_index]

        iteration += 1
        if iteration > max_iteration:
            print('Reconstruction halted by Max iter')
            break

    reconstructed_x = pca_model.predict(x_copy)

    return reconstructed_x

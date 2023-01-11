# -*- coding: utf-8 -*-
import numpy as np


def calculate_anomaly_score(x_data, pca_info):
    t2 = calculate_pca_t2(pca_info=pca_info, data=x_data)
    spe = calculate_pca_spe(pca_info=pca_info, data=x_data)
    return float(t2), float(spe)


def calculate_pca_t2(pca_info: dict, data: np.ndarray):
    lv_num = pca_info['lv_num']
    x_mean = np.array(pca_info['x_mean'])
    x_std = np.array(pca_info['x_std'])
    t_std = pca_info['t_std']
    loading_p = pca_info['loading_p']

    data_norm = (data - x_mean) / x_std

    t_score = np.dot(data_norm, loading_p.T)

    T2_df = (t_score / t_std) * (t_score / t_std)

    T2 = 0
    for i in range(lv_num):
        T2 += T2_df[:, i]
    return T2


def calculate_pca_spe(pca_info, data):
    x_num = pca_info['x_num']
    x_mean = np.array(pca_info['x_mean'])
    x_std = np.array(pca_info['x_std'])
    loading_p = pca_info['loading_p']

    x_norm = (data - x_mean) / x_std

    t_score = np.dot(x_norm, loading_p.T)
    x_hat_norm = np.dot(t_score, loading_p)

    x_error = x_norm - x_hat_norm
    sum_x_error = x_error * x_error

    spe = 0

    for i in range(x_num):
        spe += sum_x_error[:, i]
    return spe

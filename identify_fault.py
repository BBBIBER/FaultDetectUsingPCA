# -*- coding: utf-8 -*-
import numpy as np


def calculate_contribution(model_info: dict,
                           group_target: np.ndarray, group_reference: np.ndarray = None,
                           calc_type: str = "Weight 1") -> list:
    """잠재변수 모델의 공헌도를 계산한다.

    Args:
        model_info: 모델에 사용된 정보로 이루어진 Dictionary
        group_target: 타겟 데이터
        group_reference: 비교 데이터
        calc_type: 공헌도 조사 계산 방식(Weight 1, Normalized)

    Returns:
        contributions: [변수 별 공헌도 값]
    """
    x_num = model_info['x_num']
    t_std = model_info['t_std']
    model_x_mean = model_info['x_mean']
    model_x_std = model_info['x_std']

    # PCA, PLS 판단
    model_type, weight_key = _judge_model_type(model_info=model_info)

    weight_p_w = model_info[weight_key]
    group_target_mean = _get_average(selected=group_target)
    group_target_scaled = _scale_x(model_x_mean=model_x_mean, model_x_std=model_x_std, data=group_target_mean)

    if group_reference is None:
        group_reference_mean = [0 for i in range(x_num)]
        group_reference_scaled = [0 for i in range(x_num)]
    else:
        group_reference_mean = _get_average(selected=group_reference)
        group_reference_scaled = _scale_x(model_x_mean=model_x_mean, model_x_std=model_x_std, data=group_reference_mean)

    contribution = []
    if calc_type == "Weight 1":
        # ct = (Xj1_normalized - Xj2_normalized) * Weight * Std(t1) * Std(t1)
        # ct = (Xj1_normalized - Xj2_normalized) * Loading * Std(t1) * Std(t1)

        for j in range(x_num):

            ct = (group_target_scaled[j] - group_reference_scaled[j]) * abs(weight_p_w.T[j, 0]) * t_std[0] * t_std[0]

            if ct > 10:
                ct = 10
            elif ct < -10:
                ct = -10

            contribution.append(float(ct))
        return contribution

    elif calc_type == "Normalized":
        # ct = (Xj1_normalized - Xj2_normalized)
        for j in range(x_num):
            ct = (group_target_mean[j] - group_reference_mean[j])
            contribution.append(float(ct))
        return contribution
    else:
        raise TypeError("The Calculate Type(arg: calc_type) should be one of ['Weight 1', 'Normalized']")


def _judge_model_type(model_info: dict) -> [str, str]:
    if 'weight_star' in model_info.keys():
        model_type = "PLS"
        weight_key = 'weight_star'
    else:
        model_type = "PCA"
        weight_key = 'loading_p'
    return [model_type, weight_key]


def _get_average(selected: np.ndarray) -> np.ndarray:
    g1_mean = selected.mean(axis=0)
    return g1_mean


def _scale_x(model_x_mean: np.ndarray, model_x_std: np.ndarray, data: np.ndarray):
    data_scaled = (data - model_x_mean) / model_x_std
    return data_scaled

        
def indentify_fault_variable(contribution: list, x_columns: list) -> [int, str]:
    abs_contribution = [abs(c) for c in contribution]
    where_maximum_is = abs_contribution.index(max(abs_contribution))
    fault_variable = x_columns[where_maximum_is]
    return where_maximum_is, fault_variable

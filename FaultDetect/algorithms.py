# -*- coding: utf-8 -*-
from .detect_fault import *
from .identify_fault import *
from .calculate_anomaly_score import *
from .reconstruct_fault import *


def diagnose_fault_using_pca_model(x: np.ndarray,
                                   model_info: dict, model_fault_reference: list,
                                   tag_list: list) -> [str, str, str]:
    """ 패턴 인식 기법(PCA 모델)을 이용하여 공정 이상을 감시하고 공정 이상의 원인을 파악하는 알고리즘
    1. Fault Detection
    2. Fault Sensor Identification
    Args:
        x:
        model_info:
        model_fault_reference:
        tag_list:
    Returns:
    """
    t2, spe = calculate_anomaly_score(x_data=x, pca_info=model_info)
    fault_status = detect_x_fault(t2=t2, spe=spe, limit=model_fault_reference)

    if int(fault_status) is int(0):
        detect_alarm = False
        fault_variable = None
    else:
        detect_alarm = True
        contribution: list = calculate_contribution(model_info=model_info, group_target=x, calc_type="Weight 1")
        _, fault_variable = indentify_fault_variable(contribution=contribution, x_columns=tag_list)

    return detect_alarm, fault_variable

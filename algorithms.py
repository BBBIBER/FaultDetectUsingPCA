# -*- coding: utf-8 -*-
from .detect_fault import *
from .identify_fault import *
from .calculate_anomaly_score import *


def diagnose_fault(x: np.ndarray, y_measure: float,
                   y_predict_model, model_fault_reference: float,
                   pca_model_info: dict, pca_limit: list,
                   tag_list: list, msg: dict,
                   group_reference: np.ndarray = None) -> [str, str, str]:

    t2, spe = calculate_anomaly_score(x_data=x, pca_info=pca_model_info)
    fault_status = detect_x_fault(t2=t2, spe=spe, limit=pca_limit)

    if int(fault_status) is int(0):
        detect_msg_pca: str = "No Fault"
        identify_msg_pca: str = ""
        fault_variable_pca = None
    else:
        detect_msg_pca = f"{msg['unit_name']} > {msg['tag_name']} > {msg['step_name']} > {msg['Error_name']}"

        contribution = calculate_contribution(model_info=pca_model_info,
                                              group_target=x, group_reference=group_reference,
                                              calc_type="Weight 1")

        where_x, fault_variable_pca = indentify_fault_variable(contribution, x_columns=tag_list)
        identify_msg_pca = f"{msg['unit_name']} > {msg['tag_name']} > {msg['step_name']} > {msg['Error_name']}"

        fault_variable_reconstructed = x[where_x]  # 아직 확립되지 않았으므로 통과
        x[where_x] = fault_variable_reconstructed

    y_predict = float(y_predict_model.predict(x))

    y_status = detect_anomaly_residual(ref=model_fault_reference,
                                       meas=y_measure,
                                       pred=y_predict,
                                       freeze=None)
    if int(y_status) is int(0):
        detect_msg_vs = "No Fault"
    else:
        detect_msg_vs = f"{msg['unit_name']} > {msg['tag_name']} > {msg['step_name']} > {msg['Error_name']}"

    return detect_msg_pca, identify_msg_pca, fault_variable_pca, detect_msg_vs, y_predict


def diagnose_fault_using_virtual_sensors(x: np.ndarray, y_measure: float,
                                         y_predict_model, model_fault_reference: float,
                                         model_info: dict,
                                         tag_list: list, msg: dict) -> [str, str, str]:
    """ 가상센서를 이용하여 공정 이상의 원인을 파악하는 알고리즘
    1. Predict Y Using Virtual Sensor
    2. SEVA Test
    3. Fault Sensor Identification
    Args:
        x:
        y_measure:
        y_predict_model:
        model_fault_reference:
        model_info:
        tag_list:
        msg:

    Returns:

    """
    y_predict = float(y_predict_model.predict(x))

    y_status = detect_anomaly_residual(ref=model_fault_reference,
                                       meas=y_measure,
                                       pred=y_predict,
                                       freeze=None)
    if int(y_status) is int(0):
        detect_alarm = False
        alarm_message = ""
        fault_variable = None
    else:
        detect_alarm = True

        contribution = calculate_contribution(model_info=model_info, group_target=x, calc_type="Weight 1")

        _, fault_variable = indentify_fault_variable(contribution=contribution, x_columns=tag_list)
        alarm_message = f"{msg['unit_name']} > {msg['step_name']} > {fault_variable} is Fault"

    return y_predict, detect_alarm, alarm_message, fault_variable


def diagnose_fault_using_pca_model(x: np.ndarray,
                                   model_info: dict, model_fault_reference: list,
                                   tag_list: list, msg: dict) -> [str, str, str]:
    """ 패턴 인식 기법(PCA 모델)을 이용하여 공정 이상을 감시하고 공정 이상의 원인을 파악하는 알고리즘
    1. Fault Detection
    2. Fault Sensor Identification

    Args:
        x:
        model_info:
        model_fault_reference:
        tag_list:
        msg:

    Returns:

    """
    t2, spe = calculate_anomaly_score(x_data=x, pca_info=model_info)
    fault_status = detect_x_fault(t2=t2, spe=spe, limit=model_fault_reference)

    if int(fault_status) is int(0):
        detect_alarm = False
        alarm_messsage = ""
        fault_variable = None
    else:
        detect_alarm = True

        contribution: list = calculate_contribution(model_info=model_info, group_target=x, calc_type="Weight 1")

        _, fault_variable = indentify_fault_variable(contribution=contribution, x_columns=tag_list)
        alarm_messsage = f"{msg['unit_name']} > {msg['step_name']} > {fault_variable} is Fault"

    return detect_alarm, alarm_messsage, fault_variable


def predict_y_using_virtual_sensors(x: np.ndarray,
                                    y_predict_model,
                                    model_fault_reference: float,
                                    msg: dict,
                                    y_measure=None) -> [str, str]:
    """ Virtual sensor를 이용한 분석기, 센서 이중화 실시
    분석기 고장을 감시하고 고장 시 백업
    Args:
        x:
        y_measure:
        y_predict_model:
        model_fault_reference:
        msg:

    Returns:

    """
    y_predict = float(y_predict_model.predict(x))

    if y_measure is None:
        if y_predict >= model_fault_reference:
            y_status = 1
        else:
            y_status = 0
    else:
        y_status = detect_anomaly_residual(ref=model_fault_reference, meas=y_measure, pred=y_predict, freeze=None)

    if int(y_status) is int(0):
        detect_alarm = False
        detect_msg = "No Fault"
    else:
        detect_alarm = True
        detect_msg = f"{msg['unit_name']} > {msg['step_name']} > Error"

    return detect_alarm, detect_msg, y_predict

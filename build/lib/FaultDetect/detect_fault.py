# -*- coding: utf-8 -*-
def detect_anomaly_residual(ref: float, meas: float, pred: float, freeze=None):
    """ 잔차 분석법을 이용한 고장 진단
    동결 값 입력 시 고장으로 판별함.
    Args:
        ref: 잔차 분석의 기준 값(예 3시그마: 0.795*3)
        meas: 측정 데이터
        pred: 예측 데이터
        freeze: 동결 여부

    Returns:
        0 if normal, 1 otherwise.
    """
    # 잔차 계산
    err = abs(meas - pred)

    if freeze is not None:
        # 잔차 분석
        if (err >= ref) or (freeze is True):
            return 1
        else:
            return 0
    else:
        if err >= ref:
            return 1
        else:
            return 0


def detect_x_fault(t2: float, spe: float, limit: list):
    """
    T2, SPE 값으로 Detect Fault 수행
    경계 값보다 크거나 같을 시 고장 판별
    Args:
        t2: PCA, AutoEncoder Hotelling's T2
        spe: PCA, AutoEncoder SPE(Squared Prediction Error)
        limit: Limit of T2, SPE
    Returns:
        0 if value < limit, 1 otherwise.
    """
    t2_limit = float(limit[0])
    spe_limit = float(limit[1])
    
    if (t2 >= t2_limit) or (spe >= spe_limit):
        result = 1
    else:
        result = 0
    # print(f"======================X Fault Detection has been completed.=============")
    return result

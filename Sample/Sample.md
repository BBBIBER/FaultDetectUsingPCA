<Advanced sensor validation using optimization.>

## Import Library


```python
import joblib
import numpy as np
import pandas as pd
```

## Load Data


```python
test_data = pd.read_csv('test_data.csv', index_col='DateTime', encoding='utf-8-sig')
test_x = test_data.values[:, :-1]
test_y = test_data.values[:, -1]
tag_list = test_data.columns[:-1]
x_num = len(tag_list)
```


```python
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GAS_F</th>
      <th>WTR_F</th>
      <th>WTR_T</th>
      <th>STM_T</th>
      <th>STM_P</th>
      <th>FLUE_T2</th>
      <th>FLUE_O2</th>
    </tr>
    <tr>
      <th>DateTime</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1384</th>
      <td>1637.32</td>
      <td>21.2</td>
      <td>97.6</td>
      <td>216.5</td>
      <td>20.3</td>
      <td>159</td>
      <td>2.894</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_data = test_x[0].reshape(-1, x_num)
y_data = test_y[0]
y_fault_reference = 0.05
```


```python
msg = {"unit_name": "Test Unit"}
msg['step_name'] = "Test Step"
```

# Algo 1. Diagnosis Fault and Reconstruct Using PLS Model


```python
pca_info = joblib.load('pca_info.pkl')
pls_model = joblib.load('pls_model.pkl')
pls_info = joblib.load('pls_info.pkl')
```


```python
from FaultDetect.algorithms import diagnose_fault
```


```python
detect_msg_pca, identify_msg_pca, fault_variable_pca, detect_msg_vs, y_predict = diagnose_fault(x=x_data, y_measure=y_data,
                                                                                                y_predict_model=pls_model,
                                                                                                model_fault_reference=y_fault_reference,
                                                                                                pca_model_info=pca_info, pca_limit=pca_info['limit'],
                                                                                                tag_list=tag_list, 
                                                                                                msg=msg,
                                                                                                group_reference=None)
detect_msg_pca, identify_msg_pca, fault_variable_pca, detect_msg_vs, y_predict
```

    Reconstruct X: [1637.32] ---> [1346.10689139]
    Y Error- Y Predict:2.944, Y Measured:2.894
    




    ('Test Unit > Test Step > Fault Detect',
     'Test Unit > Test Step > GAS_F Error',
     'GAS_F',
     'Test Unit > Test Step > Y Error',
     2.9440021661744207)



# Algo 2. Diagnosis Fault Using PLS Model


```python
pls_model = joblib.load('pls_model.pkl')
pls_info = joblib.load('pls_info.pkl')
```


```python
from FaultDetect.algorithms import diagnose_fault_using_virtual_sensors
```


```python
y_predict, detect_alarm, alarm_message, fault_variable = diagnose_fault_using_virtual_sensors(x=x_data,
                                                                                              y_measure=y_data,
                                                                                              y_predict_model=pls_model, 
                                                                                              model_fault_reference=y_fault_reference,
                                                                                              model_info=pls_info,
                                                                                              tag_list=tag_list,
                                                                                              msg=msg)
y_predict, detect_alarm, alarm_message, fault_variable
```




    (2.823457885517407, True, 'Test Unit > Test Step > GAS_F is Fault', 'GAS_F')



# Algo 3. Diagnosis Fault Using PCA Model
1. Fault Detect
2. Fault Identfication


```python
pca_info = joblib.load('pca_info.pkl')
```


```python
from FaultDetect.algorithms import diagnose_fault_using_pca_model
```


```python
detect_alarm, alarm_messsage, fault_variable = diagnose_fault_using_pca_model(x=x_data, 
                                                                              model_info=pca_info, 
                                                                              model_fault_reference=pca_info['limit'],
                                                                              tag_list=tag_list, msg=msg)
print(detect_alarm, alarm_messsage, fault_variable)
```

    True Test Unit > Test Step > GAS_F is Fault GAS_F
    

# Algo 4. 


```python
pls_model = joblib.load('pls_model.pkl')
```


```python
from FaultDetect.algorithms import predict_y_using_virtual_sensors
```


```python
detect_alarm, detect_msg, y_predict = predict_y_using_virtual_sensors(x=x_data,
                                                                      y_predict_model=pls_model,
                                                                      model_fault_reference=y_fault_reference,
                                                                      msg=msg,
                                                                      y_measure=y_data)
detect_alarm, detect_msg, y_predict
```




    (True, 'Test Unit > Test Step > Error', 2.823457885517407)




```python

```

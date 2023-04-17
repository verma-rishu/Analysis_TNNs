Number of epochs = 200

Lags = 8

## EvolveGCN-O
| Learning Rate | Activation Function | Optimizer | MSE | Comments|
| --- | ---------- | ---- | --- | --- | 
| 0.1 | ReLU | Adam |  0.889|  
| 0.01 | ReLU | Adam | 0.976|
| 0.001 | ReLU |Adam | 0.865|
| 0.0001 | ReLU |Adam |  0.735|
| 0.1 | tanh | Adam | 0.924 |  
| 0.01 | tanh | Adam | 0.886 |
| 0.001 | tanh |Adam | 0.900|
| 0.0001 | tanh |Adam | 0.538 |
| 0.0001 | tanh |SGD |0.586|
| 0.0001 | tanh |SGDM(0.5) | 1.171|
| 0.0001 | tanh |SGDM(0.9) | 0.840| Increasing LR increase MSE|
| 0.0001 | tanh |RMSProp | 0.6779|| 


## EvolveGCN-H
| Learning Rate | Activation Function | Optimizer | MSE | Comments |
| --- | ---------- | ---- | --- | --- | 
| 0.1 | ReLU | Adam |  0.959|  
| 0.01 | ReLU | Adam | 0.979|
| 0.001 | ReLU |Adam | 1.118|
| 0.0001 | ReLU |Adam |  1.194|
| 0.1 | tanh | Adam | 1.351 |  
| 0.01 | tanh | Adam |1.275 |
| 0.001 | tanh |Adam | 0.808|
| 0.0001 | tanh |Adam |0.841  |
| 0.001 | tanh | SGD | 1.037 |
| 0.001 | tanh | SGDM(0.5) | 0.775|
| 0.001 | tanh | SGDM(0.9) | 1.119|
| 0.001 | tanh | RMSProp |0.8834  | Incresing LR increase MSE

## GConvGRU
| Learning Rate | Activation Function | Optimizer | MSE |
| --- | ---------- | ---- | --- | 
| 0.1 | ReLU | Adam |  0.883|  
| 0.01 | ReLU | Adam | 0.943|
| 0.001 | ReLU |Adam | 0.917|
| 0.0001 | ReLU |Adam |  0.701|
| 0.1 | tanh | Adam | 0.935  |  
| 0.01 | tanh | Adam | 0.914 |
| 0.001 | tanh |Adam | 0.900 |
| 0.0001 | tanh |Adam | 0.616 |
| 0.0001 | tanh | SGD | 0.808 |
| 0.0001 | tanh | SGDM(0.5) | 1.041 |
| 0.0001 | tanh | SGDM(0.9) | 0.838 |
| 0.0001 | tanh | RMSProp|0.6109 |

## GConvLSTM
| Learning Rate | Activation Function | Optimizer | MSE |
| --- | ---------- | ---- | --- | 
| 0.1 | ReLU | Adam |  0.744|  
| 0.01 | ReLU | Adam | 0.734|
| 0.001 | ReLU |Adam | 0.836|
| 0.0001 | ReLU |Adam |  0.799|
| 0.1 | tanh | Adam | 0.634 |  
| 0.01 | tanh | Adam | 0.716 |
| 0.001 | tanh |Adam | 0.849|
| 0.0001 | tanh |Adam | 0.802 |
| 0.1  | tanh | SGD | 0.967 |
| 0.1  | tanh | SGDM(0.5) | 1.063 |
| 0.1  | tanh | SGDM(0.9) | 0.673 |
| 0.1  | tanh | RMSProp| 0.6407|

![Alt text](/plots/MSE_vs_LR_ReLU.png "MSE vs Learning Rate (ReLU)")
![Alt text](/plots/MSE_vs_LR_Tanh.png "MSE vs Learning Rate (TanH)")



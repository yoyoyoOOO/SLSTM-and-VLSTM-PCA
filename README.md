# SLSTM and VLSTM-PCA

## About the project
For voids that retain their ellipsoidal characteristics during growth, the ellipsoidal void Semiaxes Long ShortTerm Memory (SLSTM) method is proposed, using the 3 principal features of the ellipsoid to represent the voids. For voids that undergo arbitrary shape changes during growth, an innovative deep learning method called Voronoi tessellation-assisted LSTM (VLSTM) is proposed. This method uses the Voronoi algorithm to standardize data features and employs Principal Component Analysis (PCA) to perform data compression before neural network training. This new method combines the Voronoi algorithm, LSTM neural networks, and PCA algorithms, and is termed as VLSTM-PCA.

## Getting start
Programming language: Python (3.8 or higher)

## Software requirement
#### os
#### time
#### math
#### sklearn
#### numpy
#### pandas
#### Tensorflow (2.10.0 or higher)
#### Keras (2.10.0 or higher)

## Dataset
The data was not disclosed because the 168 RVE models and the overall data after data processing exceed 500 GB therefore the data was not disclosed. Interested readers can contact the author.The author have placed the two trained models SLSTM_MODEL.h5 and VLSTM-PCA_MODEL.h5 in folders for interested readers to investigate

## Numerical simulation data preprocessing
Numerical simulation data preprocessing method is provided in Numerical simulation data preprocessing method.py

## SLSTM
The SLSTM network structure was shown in Fig. The corresponding method was provided in the SLSTM network structure.py. The method to obtain SLSTM data features was provided in the Method to obtain semiaxes of ellipsoidal voids.py
![image text](https://github.com/yoyoyoOOO/SLSTM-and-VLSTM-PCA/blob/main/VLSTM-PCA%26SLSTM/Picture/SLSTM.png)

## VLSTM-PCA
The VLSTM-PCA network structure was shown in Fig. The corresponding method was provided in the VLSTM-PCA network structure.py. The method to obtain Voronoi data was provided in the Voronoi tessellation for inverse operator.py
![image text](https://github.com/yoyoyoOOO/SLSTM-and-VLSTM-PCA/blob/main/VLSTM-PCA%26SLSTM/Picture/VLSTM-PCA.png)

## SLSTM result

![image text](https://github.com/yoyoyoOOO/SLSTM-and-VLSTM-PCA/blob/main/VLSTM-PCA%26SLSTM/Picture/SLSTM%20original%20dataset.png)
### <p align="center">Original dataset</p>
![image text](https://github.com/yoyoyoOOO/SLSTM-and-VLSTM-PCA/blob/main/VLSTM-PCA%26SLSTM/Picture/SLSTM%20recurrent%20prediction%20.png)
### <p align="center"Recurrent prediction</p>

## VLSTM-PCA result

![image text](https://github.com/yoyoyoOOO/SLSTM-and-VLSTM-PCA/blob/main/VLSTM-PCA%26SLSTM/Picture/VLSTM-PCA%20original%20dataset.png)
### <p align="center"Original dataset</p>
![image text](https://github.com/yoyoyoOOO/SLSTM-and-VLSTM-PCA/blob/main/VLSTM-PCA%26SLSTM/Picture/VLSTM-PCA%20original%20dataset%20after%20PCA.png)
### <p align="center"Original dataset after PCA</p>
![image text](https://github.com/yoyoyoOOO/SLSTM-and-VLSTM-PCA/blob/main/VLSTM-PCA%26SLSTM/Picture/VLSTM-PCA%20recurrent%20prediction%20.png)
### <p align="center"Recurrent prediction</p>



# Nitrate Prediction Using Stacked Random Forest

Introduction
This repository contains the code for predicting nitrate concentrations using a stacked random forest model. The model leverages multiple random forest regressors to improve prediction accuracy. The dataset used includes features such as SST (Sea Surface Temperature), SSS (Sea Surface Salinity), MLD (Mixed Layer Depth), and Chla (Chlorophyll a).
Prerequisites
Before running the code, ensure you have the following libraries installed:
numpy
pandas
scikit-learn
matplotlib
joblib
You can install these libraries using pip:
pip install numpy pandas scikit-learn matplotlib joblib

Dataset
The dataset used in this project should be in an Excel file and include the following columns:
Nitrate: The target variable.
SST: Sea Surface Temperature.
SSS: Sea Surface Salinity.
MLD: Mixed Layer Depth.
Chla: Chlorophyll a.
Replace your_path in the code with the actual path to your dataset file.
Code Structure
Data Preparation: The data is split into training and testing sets.
Model Training: Two random forest regressors are trained. The first model (rf1) is trained on the original features, and its predictions are used as an additional feature for the second model (rf2).
Hyperparameter Tuning: Grid search is used to find the best hyperparameters for both models.
Model Evaluation: The models are evaluated using mean squared error and visualized using 2D histograms.
Independent Validation: The final model is validated on an independent dataset.


Visualization
The code includes three visualization plots:
Model Training: A 2D histogram showing the relationship between field nitrate and modeled nitrate for the training set.
Model Testing: A 2D histogram showing the relationship between field nitrate and modeled nitrate for the testing set.
Independent Validation: A 2D histogram showing the relationship between field nitrate and modeled nitrate for the independent validation set.

Contact
For any questions or issues, please contact tingyugu@zju.edu.cn.
License
This project is licensed under the MIT License - see the LICENSE file for details.






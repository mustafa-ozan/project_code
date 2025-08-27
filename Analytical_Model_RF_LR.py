# project 2D Version of single bacterium motion in one stable transmitter
# one stable receiver
# Analytical model generation using LR RF
#
#
# programmer: MUSTAFA OZAN DUMAN
#
# 25.06.2025

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

X_train = np.load('X_train_data_original.npy')
y_train = np.load('y_train_data_original.npy')

X_test = np.load('X_test_data_original.npy')
y_test = np.load('y_test_data_original.npy')

##### ML models RF and LR

input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

# Fit scalers on training data and transform
X_train_scaled = input_scaler.fit_transform(X_train)
y_train_scaled = output_scaler.fit_transform(y_train)

# Transform validation data using the *fitted* scalers
X_test_scaled = input_scaler.transform(X_test)
y_test_scaled = output_scaler.transform(y_test)

# RF model
rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train_scaled, y_train_scaled)
y_pred_rf = rf_model.predict(X_test_scaled)

test_pred_original_scale_rf = output_scaler.inverse_transform(y_pred_rf)

r2_rf_out_1 = r2_score(y_test[:, 0], test_pred_original_scale_rf[:, 0])
mse_rf_out_1 = mean_squared_error(y_test[:, 0], test_pred_original_scale_rf[:, 0])

r2_rf_out_2 = r2_score(y_test[:, 1], test_pred_original_scale_rf[:, 1])
mse_rf_out_2 = mean_squared_error(y_test[:, 1], test_pred_original_scale_rf[:, 1])

# LR model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train_scaled)
y_pred_lr = lr_model.predict(X_test_scaled)

test_pred_original_scale_lr = output_scaler.inverse_transform(y_pred_lr)

r2_lr_out_1 = r2_score(y_test[:, 0], test_pred_original_scale_lr[:, 0])
mse_lr_out_1 = mean_squared_error(y_test[:, 0], test_pred_original_scale_lr[:, 0])

r2_lr_out_2 = r2_score(y_test[:, 1], test_pred_original_scale_lr[:, 1])
mse_lr_out_2 = mean_squared_error(y_test[:, 1], test_pred_original_scale_lr[:, 1])


# Print coefficients and intercepts for each output
print("Linear Regression Coefficients:")
for i in range(y_train.shape[1]):
    print(f"Output {i + 1}:")
    print("  Coefficients:", lr_model.coef_[i])
    print("  Intercept:", lr_model.intercept_[i])


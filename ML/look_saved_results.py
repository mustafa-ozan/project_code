import numpy as np

epoch_losses = np.load('epoch_losses_best_case.npy')
val_losses = np.load('val_losses_best_case.npy')

params = np.load('best_params.npy', allow_pickle=True)
print("Loaded params:", type(params))
print(params)

X_train_data_original = np.load('X_train_data_original.npy')
y_train_data_original = np.load('y_train_data_original.npy')
X_val_data_original = np.load('X_val_data_original.npy')
y_val_data_original = np.load('y_val_data_original.npy')
X_test_data_original = np.load('X_test_data_original.npy')
y_test_data_original = np.load('y_test_data_original.npy')

r2_scores = np.load('r2_scores_best_case.npy')
mse_scores = np.load('mse_scores_best_case.npy')


# Helper function to print info
def print_array_info(name, array, max_elements=10):
    print(f"\n{name}:")
    print(f"  Type: {type(array)}")
    print(f"  Shape: {array.shape}")
    print(f"  Dtype: {array.dtype}")

    if array.size <= max_elements:
        print("  Values:", array)
    else:
        print(f"  First {max_elements} values: {array[:max_elements]}")


# Print info for each array
print_array_info("epoch_losses", epoch_losses)
print_array_info("val_losses", val_losses)
print_array_info("X_train_data_original", X_train_data_original)
print_array_info("y_train_data_original", y_train_data_original)
print_array_info("X_val_data_original", X_val_data_original)
print_array_info("y_val_data_original", y_val_data_original)
print_array_info("X_test_data_original", X_test_data_original)
print_array_info("y_test_data_original", y_test_data_original)
print_array_info("r2_scores", r2_scores)
print_array_info("mse_scores", mse_scores)

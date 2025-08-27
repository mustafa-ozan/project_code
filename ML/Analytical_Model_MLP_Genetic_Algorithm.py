# project 2D Version of single bacterium motion in one stable transmitter
# one stable receiver
# Analytical model generation using MLP
# Genetic Algorithm code for MLP hyperparameter optimization
#
#
# programmer: MUSTAFA OZAN DUMAN
#
# 25.06.2025



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import random
import copy
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.container')
warnings.filterwarnings("ignore", message="The input matches the zero point of the quantized input.")

# Load and split the data
file_path = 'bacteria_2D_results_zeros_extracted.xlsx'
df = pd.read_excel(file_path)

input_cols = df.columns[:3].tolist()
output_cols = df.columns[-2:].tolist()
df_cleaned = df.dropna(subset=output_cols)

X = df_cleaned[input_cols]
y_all_outputs = df_cleaned[output_cols]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_all_outputs, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=42
)

print(f"Total data points: {len(df)}")
print(f"Training data points: {len(X_train)}")
print(f"Validation data points: {len(X_val)}")
print(f"Test data points: {len(X_test)}")


##### ML models RF and LR

# input_scaler = MinMaxScaler()
# output_scaler = MinMaxScaler()
#
# # Fit scalers on training data and transform
# X_train_scaled = input_scaler.fit_transform(X_train)
# y_train_scaled = output_scaler.fit_transform(y_train)
#
# # Transform validation data using the *fitted* scalers
# X_test_scaled = input_scaler.transform(X_test)
# y_test_scaled = output_scaler.transform(y_test)
#
# # RF model
# rf_model = RandomForestRegressor(random_state=42)
#
# rf_model.fit(X_train_scaled, y_train_scaled)
# y_pred_rf = rf_model.predict(X_test_scaled)
#
# test_pred_original_scale_rf = output_scaler.inverse_transform(y_pred_rf)
#
# r2_rf_out_1 = r2_score(y_test[:, 0], test_pred_original_scale_rf[:, 0])
# mse_rf_out_1 = mean_squared_error(y_test[:, 0], test_pred_original_scale_rf[:, 0])
#
# r2_rf_out_2 = r2_score(y_test[:, 1], test_pred_original_scale_rf[:, 1])
# mse_rf_out_2 = mean_squared_error(y_test[:, 1], test_pred_original_scale_rf[:, 1])
#
# # LR model
# lr_model = LinearRegression()
# lr_model.fit(X_train_scaled, y_train_scaled)
# y_pred_lr = lr_model.predict(X_test_scaled)
#
# test_pred_original_scale_lr = output_scaler.inverse_transform(y_pred_lr)
#
# r2_lr_out_1 = r2_score(y_test[:, 0], test_pred_original_scale_lr[:, 0])
# mse_lr_out_1 = mean_squared_error(y_test[:, 0], test_pred_original_scale_lr[:, 0])
#
# r2_lr_out_2 = r2_score(y_test[:, 1], test_pred_original_scale_lr[:, 1])
# mse_lr_out_2 = mean_squared_error(y_test[:, 1], test_pred_original_scale_lr[:, 1])


# Define the One-Hidden-Layer MLP Model in PyTorch
class OneHiddenLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_nodes, output_size=1, dropout_rate=0.0, activation_fn=nn.ReLU):
        super(OneHiddenLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_nodes)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation_fn()
        self.output_layer = nn.Linear(hidden_nodes, output_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


def train_and_evaluate_mlp(
        params,
        X_train_data_original, y_train_data_original,  # Expecting original (unscaled) data
        X_val_data_original, y_val_data_original,  # Expecting original (unscaled) data
        X_test_data_original, y_test_data_original,  # Expecting original (unscaled) data
        input_scaler, output_scaler  # Scaler instances passed
):
    # Fit scalers on training data and transform
    X_train_scaled = input_scaler.fit_transform(X_train_data_original)
    y_train_scaled = output_scaler.fit_transform(y_train_data_original)

    # Transform validation data using the *fitted* scalers
    X_val_scaled = input_scaler.transform(X_val_data_original)
    y_val_scaled = output_scaler.transform(y_val_data_original)

    # Transform validation data using the *fitted* scalers
    X_test_scaled = input_scaler.transform(X_test_data_original)
    y_test_scaled = output_scaler.transform(y_test_data_original)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    # Get output size (number of targets)
    output_size = y_train_data_original.shape[1]  # Use original data shape for output size

    # Instantiate the model
    model = OneHiddenLayerMLP(
        input_size=X_train_tensor.shape[1],
        hidden_nodes=params['nodes'],
        output_size=output_size,
        dropout_rate=params['dropout_rate'],
        activation_fn=params['activation']
    )

    criterion = nn.MSELoss()

    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    # Early stopping params
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = params.get('patience', 10)
    best_model_state = None  # Initialize to None

    for epoch in range(params['epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Save best model weights
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # If training finished without improving, or if patience was reached,
    # ensure best_model_state is not None (e.g., if patience was 0 or 1 and no improvement from start)
    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_state)

    # Final prediction on test set
    model.eval()
    with torch.no_grad():
        test_pred_scaled_tensor = model(X_test_tensor)

    # Inverse transform predictions and ground truth
    test_pred_original_scale = output_scaler.inverse_transform(test_pred_scaled_tensor.numpy())
    y_test_original_scale = output_scaler.inverse_transform(y_test_tensor.numpy())

    # Compute MSE for multi-output (mean of MSE for both outputs)
    mse = mean_squared_error(y_test_original_scale, test_pred_original_scale, multioutput='uniform_average')

    return mse  # Return MSE to minimize


# Genetic Algorithm for Hyperparameter Optimization

class GeneticAlgorithm:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, population_size=10, generations=5):  # Remove scalers from here
        self.X_train = X_train  # Store original unscaled data
        self.y_train = y_train  # Store original unscaled data
        self.X_val = X_val  # Store original unscaled data
        self.y_val = y_val  # Store original unscaled data
        self.X_test = X_test  # Store original unscaled data
        self.y_test = y_test  # Store original unscaled data
        self.population_size = population_size
        self.generations = generations

        # Define the search space for hyperparameters
        self.param_space = {
            'nodes': list(range(1, 30)),
            'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4],
            'activation': [
                nn.ReLU,
                nn.LeakyReLU,
                nn.ELU,
                nn.Sigmoid,
                nn.Tanh,
                nn.PReLU,
                nn.SELU,
                nn.GELU,
                nn.Softplus,
                nn.Hardtanh
            ],
            'optimizer': ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'ASGD'],
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
            'batch_size': [16, 32, 64, 128],
            'epochs': [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
            'l2_reg': [0.0, 1e-5, 1e-4, 1e-3],
            'patience': [5, 10, 15, 20, 25, 30]
        }

    def create_individual(self):
        # Create a random individual (a set of hyperparameters)
        individual = {key: random.choice(values) for key, values in self.param_space.items()}
        return individual

    def mutate(self, individual):
        # Mutate one or two hyperparameters randomly
        keys = list(self.param_space.keys())
        for _ in range(random.randint(1, 2)):
            key = random.choice(keys)
            individual[key] = random.choice(self.param_space[key])
        return individual

    def crossover(self, parent1, parent2):
        # Single-point crossover
        child = {}
        keys = list(self.param_space.keys())
        crossover_point = random.randint(1, len(keys) - 1)
        for i, key in enumerate(keys):
            if i < crossover_point:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def fitness(self, individual):
        # Create new scaler instances for each fitness evaluation to ensure independent scaling
        # for each model (individual) in the GA. This is crucial.
        input_scaler_for_eval = MinMaxScaler()
        output_scaler_for_eval = MinMaxScaler()

        try:
            fitness_val = train_and_evaluate_mlp(
                individual,
                self.X_train, self.y_train,  # Pass original unscaled data
                self.X_val, self.y_val,  # Pass original unscaled data
                self.X_test, self.y_test,  # Pass original unscaled data
                input_scaler_for_eval, output_scaler_for_eval  # Pass new scaler instances
            )
        except Exception as e:
            print(f"Training failed for params {individual} with error: {e}")
            fitness_val = float('inf')  # Penalize failure
        return fitness_val

    def run(self):
        # Initialize population
        population = [self.create_individual() for _ in range(self.population_size)]

        best_individual = None
        best_fitness = float('inf')  # We are minimizing MSE

        for gen in range(self.generations):
            print(f"Generation {gen + 1}/{self.generations}")

            # Evaluate population fitness
            fitness_scores = []
            for indiv in population:
                score = self.fitness(indiv)
                fitness_scores.append(score)

                if score < best_fitness:
                    best_fitness = score
                    best_individual = indiv
                # print(f"Individual: {indiv}, MSE: {score:.4f}") # Optional: for debugging

            print(f"Best fitness (MSE) in this generation: {best_fitness:.4f}")

            # Selection: top 50% individuals survive (lower MSE is better)
            sorted_pop_with_scores = sorted(zip(fitness_scores, population), key=lambda pair: pair[0])
            population = [x for score, x in sorted_pop_with_scores[:self.population_size // 2]]

            # Generate new individuals via crossover and mutation
            children = []
            while len(children) < self.population_size - len(population):
                # Ensure there are at least two parents for crossover
                if len(population) < 2:
                    # If population is too small, create more random individuals
                    child = self.create_individual()
                else:
                    parent1, parent2 = random.sample(population, 2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                children.append(child)

            population.extend(children)

        # After all generations, return the best individual and its best MSE
        return best_individual, best_fitness


# --- Main execution ---

# No initial scaling needed here, GeneticAlgorithm will handle it

# Run GA with original (unscaled) data
ga = GeneticAlgorithm(X_train, y_train, X_val, y_val, X_test, y_test, # Pass original data
                      population_size=50, generations=30)

best_params, best_mse_on_val = ga.run()

print("\n--- Genetic Algorithm Results ---")
print("Best hyperparameters found:")
for key, value in best_params.items():
    if key == 'activation':
        print(f"{key}: {value.__name__}")
    else:
        print(f"{key}: {value}")

print(f"Best MSE on validation set (found by GA): {best_mse_on_val:.4f}")

# --- Train Final Model on X_train_val and Evaluate on X_test ---
print("\n--- Final Model Training and Evaluation ---")


def train_and_evaluate_mlp_final(
        params,
        X_train_data_original, y_train_data_original,  # Expecting original (unscaled) data
        X_val_data_original, y_val_data_original,  # Expecting original (unscaled) data
        X_test_data_original, y_test_data_original,  # Expecting original (unscaled) data
        input_scaler, output_scaler  # Scaler instances passed
):
    # Fit scalers on training data and transform
    X_train_scaled = input_scaler.fit_transform(X_train_data_original)
    y_train_scaled = output_scaler.fit_transform(y_train_data_original)

    # Transform validation data using the *fitted* scalers
    X_val_scaled = input_scaler.transform(X_val_data_original)
    y_val_scaled = output_scaler.transform(y_val_data_original)

    # Transform validation data using the *fitted* scalers
    X_test_scaled = input_scaler.transform(X_test_data_original)
    y_test_scaled = output_scaler.transform(y_test_data_original)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    # Get output size (number of targets)
    output_size = y_train_data_original.shape[1]  # Use original data shape for output size

    # Instantiate the model
    model = OneHiddenLayerMLP(
        input_size=X_train_tensor.shape[1],
        hidden_nodes=params['nodes'],
        output_size=output_size,
        dropout_rate=params['dropout_rate'],
        activation_fn=params['activation']
    )

    criterion = nn.MSELoss()

    if params['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    elif params['optimizer'] == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    # Early stopping params
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = params.get('patience', 10)
    best_model_state = None  # Initialize to None

    epoch_losses = []
    val_losses = []

    for epoch in range(params['epochs']):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(train_loader))

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        val_losses.append(val_loss)

        # print(
        #     f"Epoch {epoch + 1}/{params['epochs']} - Training Loss: {epoch_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())  # Save best model weights
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    np.save('epoch_losses_best_case.npy', epoch_losses)
    np.save('val_losses_best_case.npy', val_losses)

    np.save('best_params.npy', params)
    np.save('X_train_data_original.npy', X_train_data_original)
    np.save('y_train_data_original.npy', y_train_data_original)
    np.save('X_val_data_original.npy', X_val_data_original)
    np.save('y_val_data_original.npy', y_val_data_original)
    np.save('X_test_data_original.npy', X_test_data_original)
    np.save('y_test_data_original.npy', y_test_data_original)


    # Plot Training and Validation Losses
    plt.plot(epoch_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # If training finished without improving, or if patience was reached,
    # ensure best_model_state is not None (e.g., if patience was 0 or 1 and no improvement from start)
    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    # Load best model weights
    model.load_state_dict(best_model_state)

    # Final prediction on test set
    model.eval()
    with torch.no_grad():
        test_pred_scaled_tensor = model(X_test_tensor)

    # Inverse transform predictions and ground truth
    test_pred_original_scale = output_scaler.inverse_transform(test_pred_scaled_tensor.numpy())
    y_test_original_scale = output_scaler.inverse_transform(y_test_tensor.numpy())

    # Compute MSE for multi-output (mean of MSE for both outputs)
    # mse = mean_squared_error(y_test_original_scale, test_pred_original_scale, multioutput='uniform_average')

    # Initialize lists to store R2 and MSE for each output
    r2_scores = []
    mse_scores = []

    # Compute R2 and MSE scores for each output column (e.g., y1 and y2)
    for i in range(y_test_original_scale.shape[1]):  # Iterate over columns (outputs)
        r2 = r2_score(y_test_original_scale[:, i], test_pred_original_scale[:, i])
        mse = mean_squared_error(y_test_original_scale[:, i], test_pred_original_scale[:, i])

        r2_scores.append(r2)
        mse_scores.append(mse)

        print(f"Output {i + 1} - R²: {r2:.4f}, MSE: {mse:.4f}")

    np.save('r2_scores_best_case.npy', r2_scores)
    np.save('mse_scores_best_case.npy', r2_scores)

    # Return the R2 and MSE scores for both outputs
    return r2_scores, mse_scores

# Create new scaler instances for each fitness evaluation to ensure independent scaling
# for each model (individual) in the GA. This is crucial.
input_scaler_for_eval = MinMaxScaler()
output_scaler_for_eval = MinMaxScaler()

try:
    r2_score_test, mse_test = train_and_evaluate_mlp_final(
        best_params,
        X_train, y_train,  # Pass original unscaled data
        X_val, y_val,  # Pass original unscaled data
        X_test, y_test,  # Pass original unscaled data
        input_scaler_for_eval, output_scaler_for_eval  # Pass new scaler instances
    )

    # Print each R² score for both outputs
    for i, r2 in enumerate(r2_score_test):
        print(f"Output {i + 1} - R² Score: {r2:.4f}")

    # Print each MSE score for both outputs
    for i, mse in enumerate(mse_test):
        print(f"Output {i + 1} - MSE: {mse:.4f}")

except Exception as e:
    print(f"Training failed for params {best_params} with error: {e}")
    fitness_val = float('inf')  # Penalize failure



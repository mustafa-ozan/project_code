Predictive Modeling of Bacteria-Based Nanonetwork Performance
This repository contains the source code for the research paper, "Predictive Modeling of Bacteria-Based Nanonetwork Performance Using Simulation-Driven Machine Learning and Genetic Algorithm Optimization," which combines biological simulations with machine learning to create a predictive analytical model for bacterial nanonetworks.

Repository Structure
This repository is organized into two main sections:

matlab_code/: Contains the MATLAB scripts used to run the simulation and generate the core dataset.

ML/: Contains the Python code and saved data for the machine learning models.

Simulation (MATLAB)
The matlab_code directory houses the simulation framework for the bacterial nanonetwork. This model simulates the chemotactic behavior of an E. coli bacterium as it navigates a 2D environment. The simulation systematically explores a wide range of parameter combinations for the chemoattractant quantity (Q), distance (d), and bacterial lifespan (t 
death
​
 ) to generate a comprehensive dataset of communication performance metrics (reach time and success rate). The results of this simulation are the foundational data for training the machine learning models.

Machine Learning (Python)
The ML directory includes the Python implementation of the machine learning models used to create the analytical framework.

MLP_GA_code.py: This script contains the code for the Multi-Layer Perceptron (MLP) model. It also integrates a Genetic Algorithm (GA) for hyperparameter tuning to optimize the neural network's architecture and performance.

LR_RF_code.py: This file includes the code for the Linear Regression (LR) and Random Forest (RF) models, which were used as a baseline comparison for the MLP.

saved_results/: This folder contains the saved results of the trained machine learning models, stored as .npy files. These files can be loaded and used to reproduce the model's predictions and performance metrics without needing to retrain the models from scratch.

How to Use
To use this repository:

Clone the repository to your local machine.

Navigate to the matlab_code directory to run the simulation and generate the raw data.

Open the ML directory to inspect or run the Python scripts for training and testing the machine learning models.

You can also load the saved .npy files to directly access the results of the trained models.

Citation
If you find this code or the research useful, please consider citing our paper.

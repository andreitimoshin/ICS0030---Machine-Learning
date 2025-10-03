from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Suppress convergence warnings for cleaner output
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#
# 1. Load the dataset
#
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

#
# 2. Hyperparameters to experiment with
#
hidden_layer_sizes_list = [(100,), (150,), (200,)]
alpha_values = [0.001, 0.0001]
activation_functions = ['relu', 'tanh', 'logistic']
solvers = ['adam', 'sgd']
learning_rates = ['constant']
pca_components_list = [100, 150]
random_states = [1, 2, 3]

#
# 3. Train and evaluate the model
#
results = defaultdict(list)  # Use a dictionary to store accuracies for each hyperparameter combination

# Loop through random states
for random_state in random_states:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Loop through PCA components
    for n_components in pca_components_list:
        # Apply PCA if components are specified
        if n_components:
            pca = PCA(n_components=n_components, random_state=random_state)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
        else:
            X_train_pca, X_test_pca = X_train, X_test

        # Loop through all hyperparameters
        for hidden_layer_sizes in hidden_layer_sizes_list:
            for alpha in alpha_values:
                for activation in activation_functions:
                    for solver in solvers:
                        for learning_rate in learning_rates:
                            # Train the model
                            mlp = MLPClassifier(
                                hidden_layer_sizes=hidden_layer_sizes,
                                alpha=alpha,
                                activation=activation,
                                solver=solver,
                                learning_rate=learning_rate,
                                max_iter=500,
                                random_state=random_state
                            )
                            mlp.fit(X_train_pca, y_train)

                            # Predict and evaluate
                            y_pred = mlp.predict(X_test_pca)
                            accuracy = accuracy_score(y_test, y_pred)

                            # Create a unique key for this parameter combination
                            key = (n_components if n_components else "No PCA",
                                   hidden_layer_sizes, alpha, activation, solver, learning_rate)

                            # Append the accuracy to the list for this key
                            results[key].append(accuracy)

                            # Print Progress
                            print(f"PCA: {n_components if n_components else 'No PCA'}, "
                                  f"Hidden Layers: {hidden_layer_sizes}, Alpha: {alpha}, "
                                  f"Activation: {activation}, Solver: {solver}, "
                                  f"LR: {learning_rate}, Random State: {random_state}, Accuracy: {accuracy:.2f}")

# Calculate mean accuracy for each combination
final_results = []
for key, accuracies in results.items():
    mean_accuracy = np.mean(accuracies)
    final_results.append({'PCA components': key[0],
                          'hidden_layer_sizes': key[1],
                          'alpha': key[2],
                          'activation': key[3],
                          'solver': key[4],
                          'learning_rate': key[5],
                          'mean_accuracy': mean_accuracy})

#
# 4. Visualize the results
#
import pandas as pd

# Convert results to DataFrame for easier analysis
results_df = pd.DataFrame(final_results)

# Sort results by accuracy
results_df = results_df.sort_values(by='mean_accuracy', ascending=False)

# Top 3 highest and lowest configurations
top_3_highest = results_df.head(3)
top_3_lowest = results_df.tail(3)

# Plot accuracy for different configurations
plt.figure(figsize=(10, 6))
x_labels = [
    f"PCA: {row['PCA components']}\n"
    f"HLS: {row['hidden_layer_sizes']}\n"
    f"Alpha: {row['alpha']}\n"
    f"Activation: {row['activation']}\n"
    f"Solver: {row['solver']}\n"
    f"LR: {row['learning_rate']}"
    for _, row in top_3_highest.iterrows()
]
plt.bar(range(len(top_3_highest)), top_3_highest['mean_accuracy'], color="green")
plt.xticks(range(len(top_3_highest)), x_labels, rotation=0, fontsize=10, ha='center')
plt.ylabel('Accuracy')
plt.title("Top 3 Highest Accuracy Configurations")
plt.ylim(0, 1)  # Accuracy is between 0 and 1
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
x_labels = [
    f"PCA: {row['PCA components']}\n"
    f"HLS: {row['hidden_layer_sizes']}\n"
    f"Alpha: {row['alpha']}\n"
    f"Activation: {row['activation']}\n"
    f"Solver: {row['solver']}\n"
    f"LR: {row['learning_rate']}"
    for _, row in top_3_lowest.iterrows()
]
plt.bar(range(len(top_3_lowest)), top_3_lowest['mean_accuracy'], color="red")
plt.xticks(range(len(top_3_lowest)), x_labels, rotation=0, fontsize=10, ha='center')
plt.ylabel('Accuracy')
plt.title("Top 3 Lowest Accuracy Configurations")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Save results to a CSV file for reporting
results_df.to_csv("neural_network_results.csv", index=False)
print("\nResults saved to 'neural_network_results.csv'")

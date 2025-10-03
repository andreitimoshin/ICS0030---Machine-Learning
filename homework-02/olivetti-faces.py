"""
Andrei Timoshin 233797IVSB, ICS0030
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Load the Olivetti faces dataset
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Create 2D PCA plot of Olivetti faces dataset
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)
print(X_r.shape[0])

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'red', 'green', 'blue', 'purple',
          'brown', 'pink', 'black']

lw = 2

for color, i in zip(colors, range(40)):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=i)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Olivetti faces dataset')
plt.show()

# Display the first 10 images to understand the dataset
fig, ax = plt.subplots(2, 5, figsize=(10, 5),
                       subplot_kw={'xticks': [], 'yticks': []},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[i // 5, i % 5].imshow(faces.images[i], cmap='bone')
plt.suptitle("First 10 Images in the Dataset")
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define PCA dimensions to experiment with
pca_dimensions = range(2, 320, 10)  # Start at 2, increase in steps of 10
results = {}

# Visualization dictionary for plotting results
visualization_data = {"PCA Dimensions": [], "Naive Bayes": [], "SVM": [], "Logistic Regression": []}

for n_components in pca_dimensions:

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Train and test Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train_pca, y_train)
    y_pred_nb = nb_model.predict(X_test_pca)
    nb_accuracy = accuracy_score(y_test, y_pred_nb)

    # Train and test SVM
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_pca, y_train)
    y_pred_svm = svm_model.predict(X_test_pca)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)

    # Train and test Logistic Regression
    lr_model = LogisticRegression(max_iter=500, random_state=42)
    lr_model.fit(X_train_pca, y_train)
    y_pred_lr = lr_model.predict(X_test_pca)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)

    # Store results
    results[n_components] = {
        "Naive Bayes": nb_accuracy,
        "SVM": svm_accuracy,
        "Logistic Regression": lr_accuracy
    }

    # Append results for visualization
    visualization_data["PCA Dimensions"].append(n_components)
    visualization_data["Naive Bayes"].append(nb_accuracy)
    visualization_data["SVM"].append(svm_accuracy)
    visualization_data["Logistic Regression"].append(lr_accuracy)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(visualization_data["PCA Dimensions"], visualization_data["Naive Bayes"], label="Naive Bayes", marker='o')
plt.plot(visualization_data["PCA Dimensions"], visualization_data["SVM"], label="SVM", marker='o')
plt.plot(visualization_data["PCA Dimensions"], visualization_data["Logistic Regression"], label="Logistic Regression", marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Classification Accuracy")
plt.title("Classifier Performance vs. PCA Components")
plt.legend()
plt.grid()
plt.show()

# Print results
print("\n--- Summary of Results ---")
for n_components, result in results.items():
    print(f"\nPCA Components: {n_components}")
    for classifier, accuracy in result.items():
        print(f"{classifier}: {accuracy:.2f}")

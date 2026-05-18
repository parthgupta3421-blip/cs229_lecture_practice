import numpy as np 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma=1.0):
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma

    # -----------------------------
    # Kernel Functions
    # -----------------------------
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def rbf_kernel(self, x1, x2):
        return np.exp(
            -self.gamma * np.linalg.norm(x1 - x2) ** 2
        )

    def kernel(self, x1, x2):
        if self.kernel_name == 'linear':
            return self.linear_kernel(x1, x2)

        elif self.kernel_name == 'rbf':
            return self.rbf_kernel(x1, x2)

    # -----------------------------
    # Training
    # -----------------------------
    def fit(self, X, y, lr=0.001, epochs=1000):

        n_samples, n_features = X.shape

        # alphas
        self.alpha = np.zeros(n_samples)

        # bias
        self.b = 0

        # Store training data
        self.X = X
        self.y = y

        # Gradient ascent on dual form
        for epoch in range(epochs):

            for i in range(n_samples):

                prediction = 0

                for j in range(n_samples):
                    prediction += (
                        self.alpha[j]
                        * y[j]
                        * self.kernel(X[j], X[i])
                    )

                prediction += self.b

                # Check constraint
                condition = y[i] * prediction

                if condition < 1:
                    self.alpha[i] += lr * (
                        1 - condition
                    )

                    # Clip alpha
                    self.alpha[i] = min(
                        self.alpha[i],
                        self.C
                    )

        # Support vectors
        sv = self.alpha > 1e-5

        self.support_vectors = X[sv]
        self.support_labels = y[sv]
        self.support_alphas = self.alpha[sv]

    # -----------------------------
    # Decision Function
    # -----------------------------
    def decision_function(self, X):

        output = []

        for x in X:

            prediction = 0

            for alpha, sv_y, sv_x in zip(
                self.support_alphas,
                self.support_labels,
                self.support_vectors
            ):

                prediction += (
                    alpha
                    * sv_y
                    * self.kernel(sv_x, x)
                )

            prediction += self.b

            output.append(prediction)

        return np.array(output)

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, X):
        return np.sign(
            self.decision_function(X)
        )
# Generate data
X, y = make_blobs(
    n_samples=200,
    centers=2,
    random_state=42
)

# Convert labels to -1 and 1
y = np.where(y == 0, -1, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create SVM
model = SVM(
    kernel='rbf',
    C=1.0,
    gamma=0.5
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)        
        
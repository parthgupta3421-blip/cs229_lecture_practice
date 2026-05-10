from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

class Guassiannaivebayes:
    def __init__(self,):
        self.classes = None 
        self.priors = {}
        self.mean = {}
        self.var = {}
        
        
    def fit (self,X,y):
        self.classes= np.unique(y)
        
        #calculate statistics for each class 
        for c in self.classes:
            X_c = X[y==c]    
            self.mean[c]=np.mean(X_c,axis=0)
            self.var[c]=np.var(X_c,axis=0)
            self.priors[c]=X_c.shape[0]/X.shape[0]
            
    def gaussian_pdf(self, class_idx, x):        
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        epsilon = 1e-9
        numerator = np.exp(-((x - mean) ** 2) / (2 * var + epsilon))

        denominator = np.sqrt(2 * np.pi * var + epsilon)

        return numerator / denominator
    
    def predict_sample(self, x):

        posteriors = []

        # Compute posterior for each class
        for c in self.classes:

            # log(P(y))
            prior = np.log(self.priors[c])

            # log(P(x1|y) * P(x2|y) * ...)
            conditional = np.sum(
                np.log(
                    self.gaussian_pdf(c, x)
                )
            )

            # log posterior
            posterior = prior + conditional

            posteriors.append(posterior)

        # Return class with highest posterior
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):

        predictions = []

        for x in X:
            predictions.append(self.predict_sample(x) )

        return np.array(predictions)

    def accuracy(self, y_true, y_pred):

        return np.mean(y_true == y_pred)




iris = load_iris()

X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = Guassiannaivebayes()

model.fit(X_train, y_train)

predictions = model.predict(X_test)


acc = model.accuracy(y_test, predictions)

print("Predictions:")
print(predictions)

print("\nActual:")
print(y_test)

print("\nAccuracy:")
print(acc)
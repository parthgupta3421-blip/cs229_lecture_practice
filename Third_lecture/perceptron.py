import numpy as np 
class Perceptron:
    
    def __init__ (self,learning_rate=0.1,epochs=10):
        self.learning_rate = learning_rate 
        self.epochs = epochs 
        
        self.weights = None 
        self.bias = None 
        
    def step_function(self,z):
        if z >= 0:
            return 1
        else:
            return 0
    
    def fit(self,X,y):
        
        n_features = X.shape[1]
        
        self.weights = np.zeros(n_features)
        self.bias = 0    
        
        for epochs in range(self.epochs):
            for i in range(len(X)):
                z=np.dot(X[i],self.weights)+self.bias

                y_hat = self.step_function(z)
                
                error = y[i] - y_hat
                
                self.weights = (self.weights + self.learning_rate*error*X[i])
                self.bias = (self.bias + self.learning_rate*error)
    
    def predict(self,x):
        z = np.dot(x,self.weights)+self.bias
        return self.step_function(z)
    
    def predict_batch(self,X):
        prediction = []
        for x in X:
            prediction.append(self.predict(x))
        return np.array(prediction)
    
#datset         
X = np.array([
    [1],
    [2],
    [3],
    [5],
    [6],
    [7]
])

y = np.array([0, 0, 0, 1, 1, 1])

model = Perceptron(learning_rate=0.1,epochs=10)

model.fit(X,y)

print("model weights:",model.weights)

print("model bias:",model.bias)    

# predictions
print(model.predict([2]))
print(model.predict([6]))

# batch prediction
print(model.predict_batch(X))
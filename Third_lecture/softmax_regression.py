import numpy as np 
from sklearn.datasets import load_iris 

class SoftmaxRegression:
    
    def __init__ (self,learning_rate=0.01,epochs=1000):
        self.learning_rate = learning_rate 
        self.epochs=epochs

        self.weights = None 
        self.bias = None 

    # softmax function 
        
    def softmax(self,z):
        z = z - np.max(z,axis=1,keepdims=True)
        exp_z = np.exp(z)
        return exp_z/np.sum(exp_z,axis=1,keepdims=True)

    # one hot encoding 
    def one_hot (self,y):
        n_classes = len(np.unique(y))
        onehot_y = np.zeros((len(y),n_classes))
        onehot_y[np.arange(len(y)),y] = 1
        return onehot_y

    #cross entropy loss
    def compute_loss(self,y_true,y_pred):
        epsilon = 1e-15
        y_pred =np.clip(y_pred,epsilon,1-epsilon)
        loss=-np.mean(np.sum(y_true*np.log(y_pred),axis=1))
        return loss
        
    #training 

    def fit (self,X,y):
        n_samples,n_features = X.shape
        n_classes = len(np.unique(y))
        
        self.weights = np.zeros((n_features,n_classes)) 
        self.bias = np.zeros((1,n_classes))
        
        y_encoded=self.one_hot(y)
        
        for epoch in range(self.epochs):
            logits = np.dot(X,self.weights) + self.bias   
            y_pred=self.softmax(logits)
            
            
            #gradients
            dw = (1/n_samples)*np.dot(X.T,(y_pred-y_encoded))
            db = (1/n_samples)*np.sum(y_pred-y_encoded,axis=0,keepdims=True)
            
            self.weights -= self.learning_rate * dw 
            self.bias -= self.learning_rate * db 
                
            if epoch %100==0:
                loss=self.compute_loss(y_encoded,y_pred)
                print(f"EPOCH{epoch},Loss:{loss:.4f}")
                
    def predict(self,X):
            logits = np.dot(X, self.weights) + self.bias
            probabilities = self.softmax(logits)
            return np.argmax(probabilities,axis=1)  
        
    def accuracy(self,y_true,y_pred):
            return np.mean(y_true==y_pred)

iris = load_iris()

X = iris.data
y = iris.target

indices = np.arange(len(X))

np.random.seed(42)
np.random.shuffle(indices)

# 80% train, 20% test
split_index = int(0.8 * len(X))

train_indices = indices[:split_index]
test_indices = indices[split_index:]

# Create train and test sets
X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)

# Scale training data
X_train = (X_train - mean) / std

# Scale test data using SAME train mean/std
X_test = (X_test - mean) / std    
                   
model = SoftmaxRegression(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = model.accuracy(y_test, predictions)

print("\nAccuracy:", accuracy)


                   
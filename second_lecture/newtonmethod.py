import numpy as np 

class LogisticRegressionNewton:
    def __init__ (self,max_iterations=100,tol=1e-6):
        self.max_iterations=max_iterations 
        self.tol=tol
        self.theta=None
        
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def fit(self,X,y):
        
        #add intercept 
        X=np.c_[np.ones(X.shape[0]),X]
        
        m,n=X.shape 
        
        self.theta= np.zeros(n)
        
        for i in range(self.max_iterations):
            z = X @ self.theta
            h = self.sigmoid(z)
            
            gradient = (1/m) * (X.T@(h-y))
            
            S=np.diag(h*(1-h))
            H= (1/m)*(X.T @ S @ X)
            
            #newton update 
            theta_new = self.theta - np.linalg.pinv(H) @ gradient 
            
            #stopping condition 
            if np.linalg.norm(theta_new - self.theta)<self.tol:
                self.theta=theta_new
                break 
            
            self.theta= theta_new 
            
    def predict_prob(self,X):
        
        X = np.c_[np.ones(X.shape[0]),X]
        
        z=X @ self.theta 
        
        return self.sigmoid(z)    
    
    def predict(self,X):
        
        probs = self.predict_prob(X)
        
        return (probs>0.5).astype(int)

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

y = np.array([0, 0, 0, 1, 1, 1])


# train model 

model = LogisticRegressionNewton()

model.fit(X,y)    
    
predictions = model.predict(X)


print(f"theta:{model.theta}")

print(f"predictions:{predictions}")

print(f"probabilities:{model.predict_prob(X)}")        
         
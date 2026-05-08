import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GDA:
    def __init__(self):
        self.mu0=None
        self.sigma=None
        self.mu1=None
        self.phi=None

    #train the model
    def fit(self,X,y):
        m,n = X.shape 
        self.phi=np.mean(y)
        
        #seprate class 
        X1= X[y==1]
        X0= X[y==0]
        
        #mean vector
        self.mu0 = np.mean(X0,axis=0)
        self.mu1 = np.mean(X1,axis=0)
        
        #cov matric 
        sigma = np.zeros((n,n))
        
        for i in range (m):
            x_i = X[i]
            if y[i]==0:
                diff = (x_i -self.mu0).reshape(-1,1)
            else:
                diff = (x_i - self.mu1).reshape(-1,1)
            sigma += diff@diff.T
        self.sigma = sigma/m 
        
    def guassian_pdf(self,X,mean):
                        
        n = X.shape[1]
        sigma_inv = np.linalg.inv(self.sigma)
        sigma_det = np.linalg.det(self.sigma)
        
        coeff= 1/(((2*np.pi)**(n/2))*(sigma_det**0.5))
        probs =  []
        for x in X:
            diff = (x-mean).reshape (-1,1)
            exponent = np.exp(-0.5*(diff.T @ sigma_inv @ diff))
            probs.append (coeff*exponent[0][0])
        return np.array(probs)
    def predict (self,X):
        p_x_y0 = self.guassian_pdf(X, self.mu0)
        p_x_y1 = self.guassian_pdf(X, self.mu1)
        # by using the bayes rule 
        p_y0 = (1-self.phi)*p_x_y0        
        p_y1 = self.phi * p_x_y1     
        return np.where (p_y0<p_y1,1,0)
    def accuracy (self,y_true,y_pred):
        return np.mean (y_true == y_pred)
    
#dataset 
data = load_breast_cancer()

X = data.data
y = data.target

#train test split     
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#feature scaling 
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#train model 
model = GDA()
model.fit(X_train,y_train)

#predictions 
predictions = model.predict(X_test)

accuracy = model.accuracy(y_test,predictions)

print ("accuracy:",accuracy)



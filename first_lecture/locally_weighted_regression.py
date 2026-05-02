import numpy as np
 
#intercet term in the algorithm
def add_intercerpt(X):
    return np.c_[np.ones((X.shape[0],1)),X]

# creates the daigonal matrix of weights 
def guassian_kernel(x_query,X,tau):
    m = X.shape[0]
    weight = np.zeros((m,m))
    for i in range(m):
        diff = x_query[1:] - X[i][1:]
        weight[i,i] = np.exp(-np.dot(diff,diff.T)/(2*tau**2))
    return weight
    
#prediction model for one point 
def lwlr_predict(x_query,X,y,tau):
    W = guassian_kernel(x_query,X,tau)
    XTWX = X.T @ W @ X
    XTWy = X.T @ W @ y
    theta = np.linalg.pinv(XTWX) @ XTWy
    return  (x_query @ theta).item()

#c prediction model for the full dataset 
def lwlr(X_train,y_train,tau):
    m = X_train.shape[0]
    y_pred = np.zeros(m)
    
    for i in range(m):
        y_pred[i]=lwlr_predict(X_train[i],X_train,y_train,tau)
    return y_pred

#now came to the datset 
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

# intercept term
X = add_intercerpt(X)

#prediction 
tau = 0.5

y_pred = lwlr(X,y,tau)

print(y_pred)    

    
    

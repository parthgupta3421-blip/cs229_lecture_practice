import numpy as np

#sigmoid function 

def sigmoid(z):
    return 1/(1+np.exp(-z))

#model prediction 
def predict(X,theta):
    return sigmoid(X @ theta)

def loss(X,y,theta):
    m = len(y)
    h = predict(X,theta)
    
    #avoid log(0)
    epsilon = 1e-9
    loss = -(1/m)*np.sum(y*np.log(h+epsilon)+(1-y)*np.log(1-h+epsilon))
    return loss 

#Gradient ascent 
def train(X,y,alpha=0.1,epoch=1000):
    m,n = X.shape
    theta = np.zeros(n)
    for i in range(epoch):
        h = predict(X,theta)
        gradient = (1/m)*X.T @ (y-h)
        
        #update gradient descent
        theta = theta + alpha * gradient 
        
        if i % 100 == 0:
            print(f"epoch:{i},loss:{loss(X,y,theta):.4f}")
    return theta        
            
#dataset
X=np.array([[1,1],[1,2],[1,3],[1,4]]) 
y=np.array([0,0,1,1])

#train model 
theta = train(X,y,alpha=0.1,epoch=1000) 
print("\nfinal theta:",theta) 
probs = predict(X,theta)
preds= (probs>=0.5).astype(int)

print("probability:",probs)
print("prediction:",preds)          
    
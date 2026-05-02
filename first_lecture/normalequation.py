import numpy as np 

x = np.array([[1,1],[1,2],[1,3],[1,4],[1,5],[1,6]])
y = np.array([[2],[4],[6],[8],[10],[12]])

def hypothesis(x,theta):
    return x @ theta 

def compute_cost(x,y,theta):
    m = len(x)
    error = x @ theta - y
    cost_function = (1/(2*m))*(error.T@error)
    return cost_function.item()

def gradient_descent(x,y,theta,alpha,iterations):
    m =  len(x)
    
    for _ in range(iterations):
        error = x @ theta - y 
        theta = theta - (alpha/m) * (x.T @ error)
     
    return theta 

theta = np.zeros((2,1))

theta = gradient_descent(x,y,theta,0.01,2000) 

print(theta)   

theta = np.linalg.inv(x.T @ x)@x.T @ y

print(theta)
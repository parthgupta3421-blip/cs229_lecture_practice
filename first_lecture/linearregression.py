import numpy as np

x = np.array([1,2,3,4,5,6])
y = np.array([2,4,6,8,10,12])

def hypothesis(x,theta0,theta1):
    return theta0 + theta1 * x

def compute_cost(x,y,theta0,theta1):
    m = len(x)
    predict = hypothesis(x,theta0,theta1)
    error = predict - y 
    cost_function = (1/(2*m))*np.sum(error**2)
    return cost_function

def gradient_descent(x,y,theta0,theta1,alpha,iterations):
    m = len (x)
    for _ in range(iterations):
        predictions = hypothesis(x,theta0,theta1)
        
        d_theta0 = (1/m)*np.sum(predictions-y)
        d_theta1 = (1/m)*np.sum((predictions-y)*x) 
        
        theta0 = theta0 - alpha*d_theta0
        theta1 = theta1 - alpha*d_theta1
        
    return theta0,theta1

theta0,theta1 = gradient_descent(x,y,0,0,0.01,5000)

 
print(f"intercept term : {theta0}, feature term : {theta1}")   


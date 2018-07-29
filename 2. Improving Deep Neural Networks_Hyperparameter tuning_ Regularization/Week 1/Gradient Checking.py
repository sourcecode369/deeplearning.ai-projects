import numpy as np 

# 1-Dimensional Gradient Checking
def forward_propagation(x, theta):
	J = np.dot(theta,x)
	return J
def backward_propagation(x,theta):
	del_theta = x
	return del_theta
# To show backward_propagation() is computing the gradiets properly
# we will have to implement gradient_checking().
# Gradient Checking is a tool that can be used to check whether 
# backward_propagation() is working correctly or not. 
def gradient_checking(x,theta,epsilon=1e-7):
	#compute gradApprox
	thetaPlus = theta + epsilon
	thetaMinus = theta - epsilon
	Jplus = forward_propagation(x,thetaPlus)
	Jminus = forward_propagation(x,thetaMinus)
	gradApprox = (Jplus - Jminus)/(2*epsilon)

	#compute grad
	grad = backward_propagation(x,theta)

	#compute difference
	numerator = np.linalg.norm(grad-gradApprox)
	denominator = (np.linalg.norm(grad) + np.linalg.norm(gradApprox))
	difference =  numerator / denominator

	#Check the Gradient
	flag=0
	if difference > 1e-7:
		print("There is an error in backward propagation and thus the Gradient is Incorrect!")
		flag = 1
	else:
		print("There are no errors and hence the Gradient is Correct!")
	return flag
 

 # N-Dimensional gradient checking
 def forward_propagation_n(X,Y,parameters):
 	m = X.shape[1]
 	W1,b1,W2,b2,W3,b3 = parameters
 	Z1 = np.dot(W1,X) + b1
 	A1 = relu(Z1)
 	Z2 = np.dot(W2,A1) + b2
 	A2 = relu(Z2)
 	Z3 = np.dot(W3,A2) + b3
 	A3 = sigmoid(Z3)

 	logprobs = np.multiply(Y,-np.log(A3)) + np.multiply((1-Y),-np.log(1-A3))
 	cost = 1/m * logprobs

 	cache = (Z1 , A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

 	return cost, cache

 def backward_propagation_n(X, Y, cache):
 	m = X.shape[1]
 	(Z1 , A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

 	dZ3 = A3 - Y
 	dW3 = 1./m * np.dot(dZ3,A2.T)
 	db3 = 1./m * np.sum(dZ3)

 	dA2 = np.dot(W3.T,dZ3)
 	dZ2 = np.multiply(dA2, np.int64(A2 > 0))
 	dW2 = 1./m * np.dot(dZ2, A1.T) * 2 
 	db2 = 1./m * np.sum(dZ2,axis=1,keepdims=True)

 	dA1 = np.dot(W2.T,dZ2)
 	dZ1 = np.multiply(dA1, np.int64(A1 > 0))
 	dW1 = 1./m * np.dot(dZ1, X.T)  
 	db1 = 4./m * np.sum(dZ1,axis=1,keepdims=True)

 	gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

def gradient_check_n(parameters,gradients, X, Y, epsilon=1e-7):
	parameters_values, _ = dictionary_to_vector(parameters)
	grad = gradients_to_vector(gradients)
	num_parameters=parameters_values.shape[0]
	J_plus = np.zeros((num_parameters,1))
	J_minus = np.zeros((num_parameters,1))
	gradapperox = np.zeros((num_parameters,1))

	for i in range(num_parameters):
		thetaPlus = np.copy(parameters_values)
		thetaPlus[i][0] = thetaPlus[i][0] + epsilon
		Jplus[i] , _= forward_propagation(X, Y, vector_to_dictionary(thetaPlus))

		thetaMinus = np.copy(parameters_values)
		thetaMinus[i][0] = thetaMinus[i][0] - epsilon 
		Jminus[i], _ = forward_propagation(X,Y,vector_to_dictionary(thetaMinus))

		gradApprox[i] = (Jplus[i] - Jminus[i]) / (2*epsilon)

	numerator = np.linalg.norm(grad-gradApprox)
	denominator = np.linalg.norm(grad) + np.linalg.norm(gradApprox)

	difference = numerator / denominator
	if difference > 1e-7:
		print("\nThere is a mistake in backward_propagation! Difference: " + str(difference))
	else: 
		print("\n Backpropagation Sucessful! Difference: " + str(difference))
	retrun difference

 	


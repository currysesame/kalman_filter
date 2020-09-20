import numpy as np
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
# https://www.youtube.com/watch?v=2-lu3GNbXM8
# https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
# https://numpy.org/doc/stable/reference/generated/numpy.array.html
measurement_true_value = range(0,100)
noise = np.random.randn(1,100)

delta_t = 0.025

Z = measurement_true_value + noise

B = np.array([[0.5*delta_t**2],
			[delta_t]])
mu = 0

X = np.array([[0],
			[0]])
P = np.array([[1, 0],
			[0, 1]])
F = np.array([[1, delta_t], 
			[0, 1]])
Q = np.array([[0.0001, 0],
			 [0, 0.0001]])
H = np.zeros((1,2))
H[0,0] = 1
H[0,1] = 0
R = 1


for i in range(100):
	X_ = np.matmul(F, X) + B * mu
	P_ = np.matmul(np.matmul(F, P), np.transpose(F)) + Q

	K = np.matmul(P_, np.transpose(H)) / (np.matmul(np.matmul(H,P_), np.transpose(H))+ R)
	X = X_ + K * (Z[0,i] - np.matmul(H, X_))
	P = P_ - K * np.matmul(H,P)

	print(X)
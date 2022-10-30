import numpy as np
from scipy import spatial

z1 = np.array([0.16, -0.26, -1.12, 0.01],)
z2 = np.array([1.08,-1.14,0.8,-0.8],)
z3 = np.array([1.94,0.08,1.83,0.17],)
z4 = np.array([-0.88,-1.01,1.36,-0.07])


# z1 = np.array([1, -1, -1, 1],)
# z2 = np.array([1,-1,1,-1],)
# z3 = np.array([1,1,1,1],)
# z4 = np.array([-1,-1,1,-1])


x = np.array([3,4,5,6])
y = np.array([4,3,2,1])

xp = np.zeros(4)
yp = np.zeros(4)
for i, z in enumerate([z1, z2, z3, z4]):
    xp[i] = np.dot(z, x) >= 0
    yp[i] = np.dot(z, y) >= 0
    # if i == 0:
    #     print('***', xp[0], yp[0])


# print(np.dot(x, y) / np.linalg.norm(x, ord=2) / np.linalg.norm(y, ord=2))
# print(xp)
# print(yp)

# h = sum(map(abs, xp - yp))
# print(h)
# dcos = spatial.distance.cosine(x, y)
# print("dcos ", dcos)
# print(1 - np.cos(np.pi * h / 4) - dcos)
import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

al1 = np.array([1,2,-3,0,1,-3])
al2 = np.array([2,1,1,1,0,2])
al3 = np.array([3,2,2,2,2,1])
al4 = np.array([2,0,3,1,-2,2])
x = np.array([1,0,1,0,1,1])

z = np.array([sigmoid(al1 @ x + 1), sigmoid(al2 @ x + 1), sigmoid(al3 @ x + 1), sigmoid(al4 @ x + 1)])
print(z)
be1 = np.array([1,2,-2,3])
be2 = np.array([2,-1,3,1])
be3 = np.array([3,1,-1,1])
import scipy

yhat = scipy.special.softmax(np.array([be1 @ z + 1, be2 @ z + 1, be3 @ z + 1]))
print(yhat)
import numpy_ml
y = [0,0,1]
print(numpy_ml.neural_nets.losses.CrossEntropy.loss(np.array([y]), np.array([yhat])))
dldb = yhat - y

import pandas
import numpy as np
import matplotlib.pyplot as plt
df = pandas.read_csv('data1.csv', header=None)
experiment = list(df.iloc[0])


def computeMLE(toss):
    return np.mean(toss)


print("MLE estimator of theta = ", computeMLE(experiment))
mles = []
n = len(experiment)
x = list(range(1, n+1))

for i in range(1, n+1):
    mles.append(computeMLE(experiment[:i]))
# print(mles)
plt.plot(x, mles)
plt.xlabel('number of experiment included')
plt.ylabel(r'$\hat \theta^{MLE}$')
plt.show()



# part 2
from scipy.stats import beta
beta_0A, beta_1A = 1005, 995
beta_0B, beta_1B = 20, 16
mean_A, _, _, _ = beta.stats(beta_0A, beta_1A, moments='mvsk')
mean_B, _, _, _ = beta.stats(beta_0B, beta_1B, moments='mvsk')
print("expected value of theta for a coin from M_A = ", mean_A)
print("expected value of theta for a coin from M_B = ", mean_B)
expectation = lambda a, b: a / (a + b)
# print(expectation(beta_0A, beta_1A), expectation(beta_0B, beta_1B))

x = np.linspace(beta.ppf(0.01, beta_0B, beta_1B),
                beta.ppf(0.99, beta_0B, beta_1B), 100)
plt.plot(x, beta.pdf(x, beta_0A, beta_1A),
       'r-', label='$M_A$ pdf')
plt.plot(x, beta.pdf(x, beta_0B, beta_1B),
       'b-', label='$M_B$ pdf')
plt.legend()
plt.show()


def computeMAP(toss, beta_0, beta_1):
    n = len(toss)
    return (toss.count(1) + beta_0 - 1) / (n + beta_1 + beta_0 - 2)


print("Assume the coin from M_A, MAP estimator of theta = ", computeMAP(experiment, beta_0A, beta_1A))
print("Assume the coin from M_B, MAP estimator of theta = ", computeMAP(experiment, beta_0B, beta_1B))

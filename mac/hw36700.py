from scipy.stats import uniform, norm
import numpy as np
a = 0
b = 1
N = 1000
ar = uniform.rvs(size=N)

integral = 0.0
f = lambda x: np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
for i in ar:
    integral += f(i)
estimation = (b-a) / N * integral
print(estimation)
# out: 0.34169109947799164

# from scipy.stats import norm
# # print(1 - norm.cdf(2))
# # # out: 0.02275013194817921

niter = 1000
n = 16
deviated = 0

for _ in range(niter):
    samples = uniform.rvs(size=n)
    if np.abs(np.mean(samples) - 0.5) > 0.5:
        deviated += 1
print("probability of mean deviation larger than 0.5 = ", deviated / niter)


class SparseMatrix:
    def __init__(self, matrix, col_wise):
        self.values, self.row_index, self.col_index = self.compress(matrix, col_wise)

    def compress(self, matrix, col_wise):
        return self.compress_col_wise(matrix) if col_wise else self.compress_row_wise(matrix)

    def compress_row_wise(self, matrix):
        values = []
        row_index = [0]
        col_index = []
        for row in range(len(matrix)):
            for col in range(len(matrix)):
                if matrix[row][col] != 0:
                    values.append(matrix[row][col])
                    col_index.append(col)
            row_index.append(len(values))
        return values, row_index, col_index


    def compress_col_wise(self, matrix):
        values = []
        row_index = []
        col_index = [0]
        for col in range(len(matrix[0])):
            for row in range(len(matrix)):
                if matrix[row][col]:
                    values.append(matrix[row][col])
                    row_index.append(row)
            col_index.append(len(values))
        return values, row_index, col_index

def multiply(mat1, mat2):
    A = SparseMatrix(mat1, False)
    B = SparseMatrix(mat2, True)

    ans = [[0] * len(mat2[0]) for _ in range(mat1)]
    for row in range(len(ans)):
        for col in range(len(ans[0])):
            mat1_row_start = A.row_index[row]
            mat1_row_end = A.row_index[row+1]

            mat2_col_start = B.row_index[col]
            mat2_col_end = B.row_index[col+1]

            while mat1_row_start < mat1_row_end and mat2_col_start < mat2_col_end:
                if A.col_index[mat1_row_start] < B.row_index[mat2_col_start]:
                    mat1_row_start += 1
                elif A.col_index[mat1_row_start] > B.row_index[mat2_col_start]:
                    mat2_col_start += 1
                else:
                    ans[row][col] += A.values[mat1_row_start] * B.values[mat2_col_start]
    return ans
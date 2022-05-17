# phone #2
# 304. Range Sum Query 2D - Immutable
# follow up, finding prefix matrix is bottleneck
# how to parallize
#

from itertools import accumulate

def prefix_sum(matrix):
    row_wise = []
    for row in matrix:
        row_wise.append(list(accumulate(row)))
    for j in range(len(matrix[0])):
        tmp = row_wise[0][j]
        for i in range(1, len(matrix)):
            tmp += row_wise[i][j]
            row_wise[i][j] = tmp
    print(row_wise)

prefix_sum([[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]])
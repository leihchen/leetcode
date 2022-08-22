from typing import *
# hchen gmail 1
# illinois 5
# redhat
# 9898
# cmu
# 163

# 1818. Minimum Absolute Sum Difference

# rotated rect sum
# https://leetcode.com/discuss/interview-question/1482150/rotatedRectSum-codesignal-Quora-Two-Sigma-OA

# count number of subarray sum equal to s, also len < k
# prefix sum + 2sum, use deque to keep index, lazy removed when index too small

#
def solution(s):
    n = len(s)
    res = 0
    for i in range(1, n):
        for j in range(i+1, n):
            a = s[:i]
            b = s[i:j]
            c = s[j:]
            if a + b != b + c and b + c != c + a and a + b != c + a:
                res += 1
    return res
# print(solution('xzxzx'))

# 0 ->, 1 |, 2 >--, 3 --> , 4  |,   5  |
#                |    |        -->   >--
def solution(directions):
    facing2idx = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    i, j = 0, 0
    facing = 0
    if directions[0][0] not in (0, 2, 5):
        return False
    n = len(directions)
    m = len(directions[0])

    def d2way(d, ind):
        if d in (0, 1):
            if ind not in (0, 1):
                return False
            return ind
        if d == 2:
            if ind == 0:
                return 1
            elif ind == 3:
                return 2
            else:
                return -1
        if d == 3:
            if ind == 3:
                return 0
            elif ind == 2:
                return 1
            else:
                return -1
        if d == 4:
            if ind == 1:
                return 0
            elif ind == 2:
                return 3
            else:
                return -1
        if d == 5:
            if ind == 0:
                return 3
            elif ind == 1:
                return 2
            else:
                return -1
        return -1

    while 0 <= i < n and 0 <= j < m:
        if i == n - 1 and j == m - 1:
            break
        facing = d2way(directions[i][j], facing)
        if facing == -1:
            return False
        di, dj = facing2idx[facing]
        i += di
        j += dj

    return 0 <= i < n and 0 <= j < m and d2way(directions[i][j], facing) != -1

directions = [[0,2,1],
              [5,4,1]]
print(solution(directions))

#
# directions:
# [[0,2,1],
#  [5,4,2]]
#
#
# directions:
# [[0,0,2,2],
#  [3,0,4,2],
#  [0,3,0,0]]
# directions:
# [[2,2,4,4],
#  [1,5,2,3],
#  [1,0,3,1],
#  [1,2,0,5],
#  [1,4,4,3],
#  [4,0,2,4],
#  [2,0,1,4],
#  [3,1,4,1]]
# def solution(a, b):
#     n = len(a)
#     diff = min([a[i+1] - a[i] for i in range(n-1)])
#     bset = set(b)
#     retval = -1
#     while diff >= 1 and diff % 2 == 0:
#         tar = a[0]
#         res = 0
#         i = 0
#         while True:
#             if i < n and tar == a[i]:
#                 res += 1
#                 i += 1
#             elif tar in bset:
#                 res += 1
#             else:
#                 break
#             tar += diff
#         tar = a[0] - diff
#         if i < n:
#             diff = diff // 2
#             continue
#         while tar in bset:
#             res += 1
#             tar -= diff
#         retval = max(retval, res)
#         diff = diff // 2
#     return retval
#
#
# def solution(matrix, queries):
#     black, white = [], []
#     n, m = len(matrix), len(matrix[0])
#     for s in range(n+m-1):
#         for i in range(n):
#             j = s - i
#             if 0 <= i < n and 0 <= j < m:
#                 if s % 2 == 1:
#                     black.append([i, j])
#                 else:
#                     white.append([i, j])
#     black.sort(key=lambda x: (matrix[x[0]][x[1]], x[0], x[1]))
#     white.sort(key=lambda x: (matrix[x[0]][x[1]], x[0], x[1]))
#     for i, j in queries:
#         ni = black[i]
#         nj = white[j]
#         mi, mj = matrix[ni[0]][ni[1]], matrix[nj[0]][nj[1]]
#         avg = (mi + mj) / 2
#         if (mi + mj) % 2 == 0:
#             matrix[ni[0]][ni[1]] = avg
#             matrix[nj[0]][nj[1]] = avg
#         elif mi > mj:
#             matrix[ni[0]][ni[1]] = math.ceil(avg)
#             matrix[nj[0]][nj[1]] = math.floor(avg)
#         else:
#             matrix[ni[0]][ni[1]] = math.floor(avg)
#             matrix[nj[0]][nj[1]] = math.ceil(avg)
#         black.sort(key=lambda x: (matrix[x[0]][x[1]], x[0], x[1]))
#         white.sort(key=lambda x: (matrix[x[0]][x[1]], x[0], x[1]))
#     return matrix
#
# a = [100, 180, 200]
# b = [102, 105, 110, 115, 120, 125, 130, 131, 135, 140, 145, 150, 155, 160, 165, 170, 173, 175, 185, 190, 195, 205, 210, 215, 220, 225, 230, 235, 240, 241, 245, 446, 471]






# A sawtooth sequence is a sequence of numbers that alternate between increasing and decreasing. In other words, each element is either strictly greater than its neighbouring elements or strictly less than its neighbouring elements.
#
# examples
#
# Given an array of integers arr, your task is to count the number of contiguous subarrays that represent a sawtooth sequence of at least two elements.
#
# Example
#
# For arr = [9, 8, 7, 6, 5], the output should be solution(arr) = 4.
#
# Since all the elements are arranged in decreasing order, it won't be possible to form any sawtooth subarrays of length 3 or more. There are 4 possible subarrays containing two elements, so the answer is 4.
#
# For arr = [10, 10, 10], the output should be solution(arr) = 0.
#
# Since all of the elements are equal, none of subarrays can be sawtooth, so the answer is 0.
#
# For arr = [1, 2, 1, 2, 1], the output should be solution(arr) = 10.
#
# All contiguous subarrays containing at least two elements satisfy the condition of problem. There are 10 possible contiguous subarrays containing at least two elements, so the answer is 10.
#
# Input/Output
#
# [execution time limit] 4 seconds (py3)
#
# [input] array.integer arr
#
# An array of integers.
#
# Guaranteed constraints:
# 2 ≤ arr.length ≤ 105,
# -109 ≤ arr[i] ≤ 109.
#
# [output] integer64
#
# Return the number of sawtooth subarrays.
# sawtooth subarray

def solution(arr):
    n = len(arr)
    start = 0
    end = 1
    res = 0
    while end < n:
        sign = arr[start] - arr[end]
        while end < n and arr[end] != arr[end - 1] and (arr[end - 1] - arr[end]) * sign > 0:
            end += 1
            sign *= -1

        length = end - start
        res += length * (length - 1) / 2
        if end < n and arr[end - 1] == arr[end]:  # avoid inf loop at equal
            start = end
        else:
            start = end - 1
        end = start + 1
    return res

# You are given an array of non-negative integers numbers. You are allowed to choose any number from this array and swap any two digits in it. If after the swap operation the number contains leading zeros, they can be omitted and not considered (eg: 010 will be considered just 10).
#
# Your task is to check whether it is possible to apply the swap operation at most once, so that the elements of the resulting array are strictly increasing.
#
# Example
#
# For numbers = [1, 5, 10, 20], the output should be solution(numbers) = true.
#
# The initial array is already strictly increasing, so no actions are required.
#
# For numbers = [1, 3, 900, 10], the output should be solution(numbers) = true.
#
# By choosing numbers[2] = 900 and swapping its first and third digits, the resulting number 009 is considered to be just 9. So the updated array will look like [1, 3, 9, 10], which is strictly increasing.
#
# For numbers = [13, 31, 30], the output should be solution(numbers) = false.
#
# The initial array elements are not increasing.
# By swapping the digits of numbers[0] = 13, the array becomes [31, 31, 30] which is not strictly increasing;
# By swapping the digits of numbers[1] = 31, the array becomes [13, 13, 30] which is not strictly increasing;
# By swapping the digits of numbers[2] = 30, the array becomes [13, 31, 3] which is not strictly increasing;
# So, it's not possible to obtain a strictly increasing array, and the answer is false.
#
# Input/Output
#
# [execution time limit] 4 seconds (py3)
#
# [input] array.integer numbers
#
# An array of non-negative integers.
#
# Guaranteed constraints:
# 1 ≤ numbers.length ≤ 103,
# 0 ≤ numbers[i] ≤ 103.
#
# [output] boolean
#
# Return true if it is possible to obtain a strictly increasing array by applying the digit-swap operation at most once, and false otherwise.
def swap2(num):
    res = []
    l = list(str(num))
    for i in range(len(l) - 1):
        for j in range(i+1, len(l)):
            l[i], l[j] = l[j], l[i]
            res.append(int(''.join(l)))
            l[i], l[j] = l[j], l[i]

    return res




# print(solution([3, 900, 10]))


# You are given a matrix of integers field of size n × m representing a game field, and also a matrix of integers figure of size 3 × 3 representing a figure. Both matrices contain only 0s and 1s, where 1 means that the cell is occupied, and 0 means that the cell is free.
#
# You choose a position at the top of the game field where you put the figure and then drop it down. The figure falls down until it either reaches the ground (bottom of the field) or lands on an occupied cell, which blocks it from falling further. After the figure has stopped falling, some of the rows in the field may become fully occupied.
#
# demonstration
#
# Your task is to find the dropping position such that at least one full row is formed. As a dropping position you should consider the column index of the cell in game field which matches the top left corner of the figure 3 × 3 matrix. If there are multiple dropping positions satisfying the condition, feel free to return any of them. If there are no such dropping positions, return -1.
#
# Note: When falling, the 3 × 3 matrix of the figure must be entirely inside the game field, even if the figure matrix is not totally occupied.


# 2 sum mod k
from collections import Counter
def solution(a, k):
    m = Counter()
    res = 0
    for elem in a:
        if k - elem % k in m:
            res += m[k - elem % k]
        if elem % k == 0 and 0 in m:
            res += m[0]
        m[elem % k] += 1
    return res


# 第二题是给一个string数组代表一串unix的指令，可能出现的指令ls,mv,cp,!index，其中index是数字,
# 代表执行index-1th的指令。例如{”ls“,"cp","mv","!3'}，那!3需要执行的就是元素commands[2]，即mv.。结果要返回ls，cp，mv分别执行多少次的数组。
def unix():
    pass



# print
def rotatedRectSum(matrix: List[List[int]], a: int, b: int) -> int:
    m = len(matrix)  # 4
    n = len(matrix[0])  # 5
    maxSum = float('-inf')
    # If we make a diagnoal line at the current index a and b represents
    # the number of element we can to include with respect to that line.
    # Since we have at least 1 element by being on an index, we subtract
    # 1 from both a and b to represent the index
    for x, y in ((a - 1, b - 1), (b - 1, a - 1)):  # (1,2), (2,1)
        # Slide through possible rectangles
        for row_min in range(0, m - x - y):  # 4 - 1 - 2 = 1
            for col_min in range(0, n - x - y): # 5 - 1 - 2 = 2
                # Find sum of rectangle
                rec_sum = matrix[row_min][col_min + x]
                corner_left = [row_min + x, col_min]
                corner_right = [row_min + y, col_min + x + y]
                j_min_step = -1
                j_max_step = 1
                # Start with top corner
                prev_index_left = prev_index_right = [row_min, col_min + x]
                for i in range(row_min + 1, row_min + x + y + 1):
                    j_min_step = j_min_step * -1 if prev_index_left == corner_left else j_min_step
                    j_max_step = j_max_step * -1 if prev_index_right == corner_right else j_max_step
                    j_min = prev_index_left[1] + j_min_step
                    j_max = prev_index_right[1] + j_max_step
                    for j in range(j_min, j_max + 1):
                        rec_sum += matrix[i][j]
                    prev_index_left = [i, j_min]
                    prev_index_right = [i, j_max]
                maxSum = rec_sum if rec_sum > maxSum else maxSum

    return maxSum
def solution(numbers):
    # sorted except one
    n = len(numbers)
    if n == 1: return True
    k = -1

    def swap2(num):
        res = []
        l = list(str(num))
        for i in range(len(l) - 1):
            for j in range(i + 1, len(l)):
                l[i], l[j] = l[j], l[i]
                res.append(int(''.join(l)))
                l[i], l[j] = l[j], l[i]

        return res
    for i in range(n - 1):
        if numbers[i] >= numbers[i + 1]:
            if k != -1:
                return False
            k = i
    if k == -1: return True
    if k == 0:
        for newnum in swap2(numbers[k]):
            if newnum < numbers[k + 1]:
                return True
    if k + 1 == n - 1:
        for newnum in swap2(numbers[k + 1]):
            if numbers[k] < newnum:
                return True
    print(swap2(numbers[k]))
    for newnum in swap2(numbers[k]):
        if numbers[k - 1] < newnum < numbers[k + 1]:
            return True
    for newnum in swap2(numbers[k + 1]):
        if numbers[k] < newnum < numbers[k + 2]:
            return True
    return False

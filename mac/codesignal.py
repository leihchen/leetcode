
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

from collections import Counter
from copy import deepcopy


def solution(arr):
    nums = []
    n = len(arr)

    def divide(arr, dall):
        if not arr:
            return
        if all(i >= 2 for i in dall.values()):
            nums.append(arr)
            return
        d = Counter()
        last = 0
        for i in range(len(arr)):
            if dall[arr[i]] == 1:
                divide(arr[last:i], d)
                last = i + 1
            elif i == len(arr) - 1 and last < i:
                nums.append(arr[last: len(arr)])
            d[arr[i]] += 1

    cnt = Counter(arr)
    divide(arr, cnt)

    def count_valid(doubles):
        res = 0
        for i in range(len(doubles)):
            for j in range(i + 1, len(doubles) + 1):
                if all(v >= 2 for v in Counter(doubles[i:j]).values()):
                    res += 1
        return res

    return sum([count_valid(num) for num in nums])

n, m = 3, 5
for s in range(n + m - 1):
    for j in range(m):
        i = s - j
        if 0 <= i < n and 0 <= j < m:
            print((i, j))
a = [["a","b","c","d"],
 ["a","c","d","e"],
 ["a","e","c","a"]]
import itertools
print(list(itertools.chain(*a)))


n,m = 4, 5
flati = 0
b = [[0 for _ in range(m)] for _ in range(n)]
for s in range(m + n - 1):
    for i in range(n):
        j = s - i
        if 0 <= i < n and 0 <= j < m:
            b[i][j] = flati + 1
            flati += 1
for row in b:
    print(row[::-1])


def solution(numbers):
    change_idx = None
    n = len(numbers)
    for i in range(1, n):
        if numbers[i] <= numbers[i-1]:
            if change_idx:
                return False
            change_idx = i-1
    if not change_idx:
        return True
    change = numbers[change_idx]
    l = list(str(change))
    ls = sorted([(l[i], i) for i in range(len(l))], key=lambda x: (x[0], -x[1]))
    l[ls[0][1]], l[ls[-1][1]] = l[ls[-1][1]], l[ls[0][1]]
    res = int(''.join(l))
    return res < numbers[change_idx+1] and (res > numbers[change_idx-1])



def maxSquareSubGrind(matrix, k):
    n = len(matrix)
    dp = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, n+1):
            dp[i][j] = matrix[i-1][j-1] + dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1]

    print(dp)
    left, right = 0, n

    def max_region(mid):
        res = 0
        for i in range(n-mid+1):
            for j in range(n-mid+1):
                res = max(res, dp[i+mid][j+mid] - dp[i][j+mid] - dp[i+mid][j] + dp[i][j])
        return res

    while left <= right:
        mid = (left + right) // 2
        curmax = max_region(mid)
        if curmax == k:
            return mid
        elif curmax > k:
            right = mid - 1
        else:
            left = mid + 1
    return right

print(maxSquareSubGrind([[1] * 4, [2] * 4, [3] * 4, [4] * 4], 20))

import math

def dist(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

def stripClosest(strip, d):
    # strip is sorted in y
    res = d
    # runtime is O(n) by math proof
    for i in range(len(strip)):
        j = i + 1
        while j < len(strip) and strip[j][1] - strip[i][1] < res:
            res = dist(strip[j], strip[i])
            j += 1
    return res

def helper(P, Q):
    # base case:
    n = len(P)
    if n <= 3:
        res = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                res = min(res, dist(P[i], P[j]))
        return res

    midx = P[n//2][0]
    left_dist = helper(P[:n//2], Q)
    right_dist = helper(P[n//2:], Q)
    cur_dist = min(left_dist, right_dist)
    stripP, stripQ = [], []
    for i in range(n):
        if abs(P[i][0] - midx) < cur_dist:
            stripP.append(P[i])
        if abs(Q[i][0] - midx) < cur_dist:
            stripQ.append(Q[i])
    stripP.sort(key=lambda x: x[1])
    sa = stripClosest(stripP, cur_dist)
    sb = stripClosest(stripQ, cur_dist)
    return min(cur_dist, sa, sb)


def closestDistOfPoint(points):
    return helper(sorted(points, key=lambda x: x[0]), sorted(points, key=lambda x: x[1]))

# test = [[2, 3], [12, 30], [40, 50], [5, 1], [12, 10], [3, 4]]
# print(closestDistOfPoint(test))

import re
import timeit
import random
import string
def suffix_prefix_3(s1, s2):
    n = min(len(s1), len(s2))
    res = 0
    for i in range(n):
        if s1[i] != s2[i]:
            break
        else:
            res = i + 1
    return res

def countDistinctSubstring(s):
    arr = list(s)
    n = len(arr)
    suffix = [''.join(arr[i:]) for i in range(n)]
    suffix.sort()
    lcp_array_sum = 0
    # print(suffix)
    for i in range(len(suffix) - 1):
        lcp_array_sum += (suffix_prefix_3(suffix[i], suffix[i+1]))
    return n*(n+1) // 2 - lcp_array_sum

def countDistinctSubstringBF(s):
    seen = set()
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            seen.add(s[i:j])
    return len(seen)

# print('===', suffix_prefix_3('ABABBAB', 'ABBAB'))
letters = string.ascii_lowercase
test1 = ''.join(random.choice(letters) for i in range(1000))
test = 'ABCD' * 500 + 'BCD' * 500
start = timeit.default_timer()
print(countDistinctSubstring("kincenvizh"))
stop = timeit.default_timer()
print('Time: ', stop - start)

start = timeit.default_timer()
print(countDistinctSubstringBF("kincenvizh"))
stop = timeit.default_timer()
print('Time: ', stop - start)

# https://www.geeksforgeeks.org/%C2%AD%C2%ADkasais-algorithm-for-construction-of-lcp-array-from-suffix-array/
# 第一题 532

# 第二题 1643
# ‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌第三题 1234
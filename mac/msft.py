from typing import *
from collections import *
import math
from bisect import bisect_left
import random
from random import randint
from copy import deepcopy
from heapq import heappush, heappop, heapify
from itertools import combinations

# https://www.1point3acres.com/bbs/thread-819025-1-1.html
# https://leetcode.com/discuss/interview-question/775767/need-help-with-this-interview-question
def minCutGcd(nums):
    n = len(nums)
    dp = [0] * n
    dp[0] = 1
    for i in range(1,n):
        dp[i] = dp[i-1] + 1
        for j in range(i):
            if math.gcd(nums[j], nums[i]) > 1:
                dp[i] = min(dp[i], (dp[j-1] + 1) if j-1 >= 0 else 1)
    return dp[-1]

# print(minCutGcd([2,3,3,2,3,3]))
# print(minCutGcd([4,5,6,7,7,9]))

# print(max_matching([[0,1],[1,3], [2,3]], 4))

def max_network_rank(starts: List[int], ends: List[int], n: int) -> int:
    adj = [0] * (n + 1)

    for a, b in zip(starts, ends):
        adj[a] += 1
        adj[b] += 1

    max_rank = 0

    for a, b in zip(starts, ends):
        max_rank = max(max_rank, adj[a] + adj[b] - 1)

    return max_rank

# print(max_network_rank([1,2,4,5], [2,3,5,6], 6))

def largest_square(a, b):
    shorter, longer = sorted((a, b))
    if longer // 4 >= shorter:
        return longer // 4
    if longer // 3 >= shorter:
        return shorter
    if shorter // 2 < longer // 3:
        return longer // 3
    return shorter // 2



# print(largest_square(10, 21))
# print(largest_square(13, 11))
# print(largest_square(2, 1))
# print(largest_square(1, 8))
# print(largest_square(5, 17))
# print(largest_square(10, 8))


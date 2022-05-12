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

print(minCutGcd([2,3,3,2,3,3]))
print(minCutGcd([4,5,6,7,7,9]))



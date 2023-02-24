from utils import *
# 1. Merging Palindromes
# Given two strings, find all palindromes that can be formed with the letters of each string.
# From those palindromes, select one from each set that,
# when combined and rearranged, produces the longest palindrome possible.
# If there are multiple palindromes of that length, choose the alphabetically smallest of them.
# Example
# $$
# \begin{aligned}
# &s 1=\text { 'aabbc' } \\
# &s 2=\text { 'ddefefq' }
# \end{aligned}
# $$
# All of the letters of the first string can make a palindrome.
# The choices using all letters are [abcba, bacab].
# All of the letters of the second string can make a palindrome.
# The choices using all letters are [defqfed, dfeqefd, edfqfde, efdqdfe, fdeqedf, fedqdef].
# The two longest results in s1 have a length of 5 .
# The six longest results in 52 have a length of 6.
# From the longest results for 51 and s2,
# merge the two that form the lowest merged palindrome, alphabetically.
# In this case, choose abcba and defqfed. The two palindromes can be combined to form a single palindrome
# if either the cor the $q$ is discarded. The alphabetically smallest combined palindrome is abdefcfedba.
# Function Description
# Complete the function mergePalindromes in the editor below. The function must return a string.
# mergePalindromes has the following parameter(s):
# string s1: a string
# string s2: a string

# sliding window,
# 2, 3, 2 ->
# break into segments

# def investablePeriods(price, p, q):
#     def partition(price):
#         pass
#     def count(nums, p, q):
#         n = len(nums)
#         pidx = [-1] + [i for i in range(n) if nums[i] == p] + [n]
#         qidx = [-1] + [i for i in range(n) if nums[i] == q] + [n]
#         res = 0
#         print(pidx, qidx)
#         for i in range(1, len(pidx) - 1):
#             for j in range(1, len(qidx) - 1):
#                 pi, qi = pidx[i], qidx[j]
#                 left = max(pidx[i - 1], qidx[j - 1])
#                 right = min(pidx[i + 1], qidx[j - 1])
#                 if pi < qi:
#                     # max(pi_1, qi_1) <- pi, qi -> min(p_i+1, q_i+1)
#                     nleft = pi - left
#                     nright = right - qi
#                     res += nleft * nright
#                 else:
#                     # max(pi_1, qi_1) <- pi, qi -> min(p_i+1, q_i+1)
#                     nleft = qi - left
#                     nright = right - pi
#                     res += nleft * nright
#         print(res)
#         return res
#     print(count([2,3,2], 2, 3))

# investablePeriods([], 2, 3)


def kUniqueMinLen(nums, k):
    cnt = Counter()
    left = 0
    res = float('inf')
    seq = []
    def remove(cnt, c):
        cnt[c] -= 1
        if cnt[c] == 0:
            del cnt[c]
    for i, num in enumerate(nums):
        cnt[num] += 1
        while len(cnt) == k:
            if i - left + 1 < res:
                res = i - left + 1
                seq = nums[left: i+1]
            remove(cnt, nums[left])
            left += 1
    print(seq)
    return res

print(kUniqueMinLen([1,1,2,3,3,3,3,2,2,4], 4))

# merge palindrome
# def
# task completion

# 0123456789
# HACKERRANK
# HACKERMAN

# def calculateDiff():
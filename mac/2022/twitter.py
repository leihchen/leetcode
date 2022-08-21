#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter, defaultdict, OrderedDict

# https://github.com/nataliekung/leetcode/tree/master/twitter-oa

# Social Network -> union find https://leetcode.com/problems/friend-circles/
# ticket buying -> https://stackoverflow.com/questions/43950000/hackerrank-buying-show-tickets-optimization
# game events -> https://leetcode.com/discuss/interview-question/375258/Twitter-or-OA-2019-or-Game-Events
# Activate fountain -> https://link.1point3acres.com/?url=https%3A%2F%2Fleetcode.com%2Fdiscuss%2Finterview-question%2F363036%2Fwalmart-oa-2019-activate-fountains
# Balanced Sales Array -> https://github.com/nataliekung/leetcode/tree/master/twitter-oa
# clostest number (if 2pass don't work)-> java treemap https://leetcode.com/discuss/interview-question/428242/audible-oa-2019-closest-numbers

# Sub Palindrome 647b
class Solution:
    def countSubstrings(self, s: str):
        n = len(s)

        def extendPalindrom(left, right):
            nonlocal res
            while left >= 0 and right < n and s[left] == s[right]:
                res.add(s[left: right+1])
                left -= 1
                right += 1

        if n == 0: return 0
        res = set()
        for i in range(n):
            extendPalindrom(i, i)
            extendPalindrom(i, i + 1)
        return list(res)
# print(Solution().countSubstrings('aabaa'))


# https://www.hackerrank.com/challenges/anagram/problem
# Complete the 'anagram' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING s as parameter.
#

def anagram(s):
    # Write your code here
    n = len(s)
    if n % 2 == 1:
        return -1
    return sum((Counter(s[:n//2]) - Counter(s[n//2:])).values())

# if __name__ == '__main__':
#     fptr = open(os.environ['OUTPUT_PATH'], 'w')
#
#     q = int(input().strip())
#
#     for q_itr in range(q):
#         s = input()
#
#         result = anagram(s)
#
#         fptr.write(str(result) + '\n')

    # fptr.close()

def weird_faculty(v):
    # na
    n = len(v)
    s = sum([1 if val else -1 for val in v])
    prefix = 0
    for i in range(n):
        if prefix > s - prefix:
            return i
        prefix += 1 if v[i] else -1
    return n
# print(weird_faculty([1, 1, 1 , 0, 1]))

def final_discounted_price(nums):
    # https://leetcode.com/submissions/detail/584833762/
    res = 0
    stack = []
    n = len(nums)
    for i in range(n)[::-1]:
        while stack and stack[-1] > nums[i]:
            stack.pop()
        if stack: res += nums[i] - stack[-1]
        else: res += nums[i]
        stack.append(nums[i])
    return res
# print(final_discounted_price([2,4,3,2,4,6]))

def auth_token(actions, ttl):
    token = {}
    for op, id, time in actions:
        if op == 0:
            token[id] = ttl + time
        elif op == 1:
            if id in token:
                exp = token[id]
                if exp >= time:
                    token[id] = ttl + time
    return sum(t >= time for t in token.values())
# print(auth_token([[0,1,1],[0,2,2],[1,1,5],[1,2,7]], 4))

def university_careerfair(events):
    res = 0
    for start, duration in events:
        if start < endtime:
            continue
        endtime = start + duration
        res += 1
    return res
# print(university_careerfair([1, 3, 4, 6], [4, 3, 3, 2]))

def even_subarray(nums):
    odd_cnt = 0
    res = 0
    n = len(nums)
    for i in range(n):
        check = set()

def card_collection(collection, d):  #->https://leetcode.com/discuss/interview-question/1536950/twitter-oa-2022-twitter-early-career-engineering-coding-challenge-questions-and-solutions.
    # if TLE
    # 第一次写用一个Hashset存Array里的所有数字.然后遍历(1, d + 1), 跳过set里面有的数字, 会有几个test
    # case
    # 超时...不知道是不是只有java会有这个问题.后面干脆不用hashset
    # 直接Arrays.sort(arrays)
    # 以后用一个pointer指向数组, 遍厉(1, d + 1)
    # 当当前数字等于pointer指向数字, pointer + + 然后到下一个loop继续检测
    collection = set(collection)
    res = []
    cost = 0
    for i in range(1, d+1):
        if i in collection: continue
        if cost + i > d: break
        cost += i
        res.append(i)
    return res
print(card_collection([1,3,4], 7))

def how_many_sentences(wordset, sentences):    # ->https://leetcode.com/discuss/interview-question/1536950/twitter-oa-2022-twitter-early-career-engineering-coding-challenge-questions-and-solutions.
    d = defaultdict(int)
    for s in wordset:
        d[''.join(sorted(s))] += 1
    res = []
    for s in sentences:
        count = 1
        for w in list(s):
            count *= d[''.join(sorted(w))]
        res.append(count)
    return res

def tallest_hashtag(postition, heights):  #-> https://leetcode.com/discuss/interview-question/373110/Twitter-or-OA-2019-or-New-Office-Design
    res = 0
    def getmax(i):
        p1, h1 = postition[i-1], heights[i-1]
        p2, h2 = postition[i], heights[i]
        tall, short = max(h1, h2), min(h1, h2)
        gap = abs(p1-p2)-1
        if tall > short + gap:
            return short + gap
        else:
            gap -= tall - short
            return tall + gap // 2 + 1 if gap % 2 == 1 else 0

    for i in range(1, len(postition)):
        if abs(postition[i] - postition[i-1]) > 1:
            res = max(res, getmax(i))
    return res
# print(tallest_hashtag([1,10],[1,51]))
# print(tallest_hashtag([1,3,7],[4,3,3]))
# print(tallest_hashtag([1,2,4,7],[4,5,7,11]))

# def tallesthastag(position,heights):
#     maxheight=0
#     for i in range(1,len(position)): # note that the length of positions start at 1
#         if abs(position[i-1]-position[i])>1: # that positions are not adjacent
#             maxheight=max(maxheight,getMaxHeight(position[i-1],position[i],heights[i-1],heights[i]))
#     return maxheight
#
# def getMaxHeight(p1,p2,h1,h2):
#     shorter=min(h1,h2)
#     taller=max(h1,h2)
#     gap=abs(p1-p2)-1
#     if taller>shorter+gap:
#         return shorter+gap
#     else:
#         diff=taller-shorter
#         gap=gap-diff
#         return int(taller+gap/2) if gap%2==0 else int(taller+gap//2+1)
#
# print(tallesthastag([1,10],[1,51]))
# print(tallesthastag([1,3,7],[4,3,3]))
# print(tallesthastag([1,2,4,7],[4,5,7,11]))

def efficient_job_processing(task, weight, runtime):  # -> https://leetcode.com/discuss/interview-question/374446/twitter-oa-2019-efficient-job-processing-service
    # dp[i][j] sum of weight of prefix task[:i] with j runtime remaining
    # dp[i][j] = max(dp[i-1][j], dp[i-1][j-task[i]*2] + weight[i])
    # dp[0][x] = 0, dp[x][0] = 0
    n = len(task)
    p = runtime // 2
    dp = [[0 for _ in range(p+1)] for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(1, 1+p):
            if j < task[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i-1][j - task[i-1]] + weight[i-1])
    print(dp)
    return max(dp[-1])
# print(efficient_job_processing([2, 2, 3, 4], [2, 2, 4, 5], 15))

def distinct_subarrays(nums, k):
    oddcnt, left = 0, 0
    res = set()
    n = len(nums)
    for right in range(n):
        if nums[right] % 2 == 1:
            oddcnt += 1
        while oddcnt > k and left < right:
            if nums[left] % 2 == 1:
                oddcnt -= 1
            left += 1
        for j in range(left, right + 1):
            res.add(repr(nums[j:right+1]))
    return len(res)
# print(distinct_subarrays([2, 2, 5, 6, 9, 2, 11, 9, 2, 11, 12], 1))
# print(distinct_subarrays([1, 3, 9, 5],2))
# print(distinct_subarrays([3, 2, 3, 2], 1))


def table_content(strs):
    res = []
    chapter, section = 1, 1
    for row in strs:
        if row[0] == row[1] == '#':
            res.append(str(chapter) + '.' + str(section) + '.' + row[2:])
            section += 1
        elif row[0] == '#':
            res.append(str(chapter) + '.' + row[2:])
            chapter += 1
            section = 1
    return '\n'.join(res)
test = ['# Algorithms This chapter covers the most basic algorithms.',
'## Sorting Quicksort is fast and widely used in practice Merge sort is a deterministic algorithm',
'## Searching REDMI NOTE 5 PRO ed graph searching algorithms MI DUAL CAMERA also used in game theory applications',
'# Data Structures This chapter is all about data structures Its a draft for now and will contain more sections in the future Binary Search Trees']
# print(table_content(test))

def find_pairs(a, b, target):  # -> https://leetcode.com/discuss/interview-question/373202/Amazon-or-OA-2019-or-Optimal-Utilization/391917
    a.sort(key=lambda x: x[1])
    b.sort(key=lambda x: x[1])
    l, r = 0, len(b) - 1
    res = []
    currmax =  float('-inf')
    while l < len(a) and r >= 0:
        curr = a[l][1] + b[r][1]
        if curr > target:
            r -= 1
        else:
            if currmax <= curr:
                if currmax < curr:
                    currmax = curr
                    res = []
                res.append([a[l][0], b[r][0]])
                i = r
                while i >= 0 and b[i][1] == b[i-1][1]:
                    res.append([a[l][0], b[i][0]])
                    i -= 1
                l += 1
    return res
# print(find_pairs([[1, 8], [2, 15], [3, 9]], [[1, 8], [2, 11], [3, 12]], 20))
# print(find_pairs( [ [1, 5], [2, 5] ], [ [1, 5], [2, 5] ], 10))


def finalInstances(instances, averageUtil):
    # Write your code here
    n = len(averageUtil)
    i = 0
    while i < n:
        if averageUtil[i] < 25 and instances > 1:
            instances = math.ceil(instances / 2)
            i += 10
        elif averageUtil[i] > 60 and instances < 10 ** 8:
            instances *= 2
            i += 10
        else:
            i += 1
    return instances
print(finalInstances(5, [6, 30, 5, 4, 8, 19, 89]))
print(math.ceil(2/2))
print('0'.isdigit())
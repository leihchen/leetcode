from typing import *
from collections import *
import math
from bisect import bisect_left, bisect_right
import random
from random import randint
from copy import deepcopy
from heapq import heappush, heappop, heapify
from itertools import combinations
from tqdm import tqdm
import numpy as np
# class Solution:
# def closestMeetingNode(edges: List[int], node1: int, node2: int) -> int:
#     # lca
#     path1 = set()
#     while node1 != -1:
#         if node1 in path1:
#             break
#         path1.add(node1)
#         node1 = edges[node1]
#     path2 = set()
#     print(path1)
#     while node2 != -1:
#         if node2 in path2:
#             break
#         path2.add(node2)
#         if node2 in path1:
#             return node2
#         node2 = edges[node2]
#     return -1
# print(closestMeetingNode([5,3,1,0,2,4,5],3,2,))
#
#
# def reverseWord(a: str, delim='_'):
#     s = a.split(delim)
#     print(s)
#     l, r = 0, len(s) - 1
#     while l < r:
#         while l < r and s[l] == '':
#             l += 1
#         while l < r and s[r] == '':
#             r -= 1
#         s[l], s[r] = s[r], s[l]
#         l += 1
#         r -= 1
#     return delim.join(s)
#
# print(reverseWord('__hot_doge_'))

# test = [[3,1], [3,3], [1,1], [1,3], [2,3], [3,2], [1,2], [2,1], [2,2]]
test = [(0,0), (0,1),(0,-1),(1,0),(1,1),(1,2),(1,-1),(1,-2),(2,1),(2,2),(2,-1),(-1,0),(-1,1),(-1,2),(-1,-1),(-1,-2),(-2,1),(-2,-1)]
x = []
y= []
# print([random.randrange(0, 2 ** 53 - 1) for _ in range(5000)])
# for i,j in test:
#     x.append(i)
#     y.append(j)
# print(x)
# print(y)
# exit(0)
# correct
def detectSquare(points):
    counter = Counter()
    res = 0
    for x1, y1 in points:
        for (x3, y3), cnt in counter.items():
            if abs(x1-x3) == 0 or abs(x1 - x3) != abs(y1 - y3):
                continue
            adding = cnt * counter[(x1, y3)] * counter[(x3, y1)]
            res += adding
            if adding > 0:
                print([(x1, y1), (x3, y3)])
        counter[(x1,y1)] += 1
    return res


def detectSquareUnAligned(points):
    seen = set()
    res = 0
    for i, (ax, ay) in enumerate(points):
        for bx, by in points:
            print((ax, ay), (bx, by))
            abx, aby = ax + bx, ay + by
            jx, jy = ay - by, bx - ax
            cx, cy = (abx + jx) / 2, (aby + jy) / 2
            dx, dy = (abx - jx) / 2, (aby - jy) / 2
            if (cx, cy) in seen and (dx, dy) in seen:
                res += 1
            print((ax, ay), (bx, by), (cx, cy), (dx, dy))
        seen.add((ax, ay))

        print(seen)
    return res
# print('---', detectSquareUnAligned([[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1]]))
# correct
# def detectRect(points):
#     counter = Counter()
#     res = 0
#     for x1, y1 in points:
#         for (x3, y3), cnt in counter.items():
#             if abs(x1-x3) == 0:
#                 continue
#             res += cnt * counter[(x1, y3)] * counter[(x3, y1)]
#         counter[(x1,y1)] += 1
#     return res


# https://leetcode.com/problems/minimum-area-rectangle-ii/
# https://leetcode.com/problems/minimum-area-rectangle-ii/discuss/980956/Python3-center-point-O(N2)
# class Solution:
#     def minAreaFreeRect(self, points: List[List[int]]) -> float:
#         ans = inf
#         seen = {}
#         for i, (x0, y0) in enumerate(points):
#             for x1, y1 in points[i+1:]:
#                 cx = (x0 + x1)/2
#                 cy = (y0 + y1)/2
#                 d2 = (x0 - x1)**2 + (y0 - y1)**2
#                 for xx, yy in seen.get((cx, cy, d2), []):
#                     area = sqrt(((x0-xx)**2 + (y0-yy)**2) * ((x1-xx)**2 + (y1-yy)**2))
#                     ans = min(ans, area)
#                 seen.setdefault((cx, cy, d2), []).append((x0, y0))
#         return ans if ans < inf else 0
def dectectRectUnAligned(points):
    res = 0
    seen = defaultdict(list)
    for i, (x0, y0) in enumerate(points):
        for x1, y1 in points[i+1:]:
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            d2 = (x0 - x1) ** 2 + (y0 - y1) ** 2
            res += len(seen[(cx, cy, d2)])
            seen[(cx, cy, d2)].append((x0, y0))

    return res

basic = [(0,1),(1,0),(0,0),(1,1)]


# print(detectSquare(test))
# print(detectSquareUnAligned([(2,6), (10,2), (8,8), (4,0),(4,10),(6,4)]))
# print(dectectRectUnAligned(test))




def terris(board):
    figure = []
    n, m = len(board), len(board[0])
    fall = []
    for col in range(m):
        tmp = []
        for row in range(n):
            tmp.append(board[row][col])
        if 'F' not in tmp:
            continue
        fs = -1
        thisfall = float('inf')
        for j, elem in enumerate(tmp):
            if elem == 'F':
                fs = j
            if elem == '#' and fs != -1:
                thisfall = min(thisfall, j - fs - 1)
                fs = -1
        if fs != -1:
            thisfall = min(thisfall, n - fs - 1)
        fall.append(thisfall)
    k = min(fall)
    for i in range(n):
        for j in range(m):
            if board[i][j] == 'F':
                figure.append([i,j])
                board[i][j] = '.'
    for x, y in figure:
        board[x+k][y] = 'F'

    return board
# test = [list('FFF'), list('.F.'), list('.FF'), list('#F.'), list('FF.'), list('...'), list('..#'), list('...')]
# for row in test:
#     print(row)
# print('-------')
# soln = terris(test)
# for row in soln:
#     print(row)
#

def lightswitch(lamps, points):
    m = 0
    for _, end in lamps:
        m = max(m, end+2)
    diff = [0] * m
    for start, end in lamps:
        diff[start] += 1
        diff[end+1] -= 1
    scan = 0
    point2light = [0] * m
    for i in range(m):
        scan += diff[i]
        point2light[i] = scan
    return [point2light[p] if p < m else 0 for p in points]

# print(lightswitch([[1,7], [5,11], [7, 9]], [7,1,5,10,9,15]))

def findranege(number, range):
    number.sort()
    idx = bisect_right(number, range[0])
    if idx > len(number) or number[-1] <= range[1] or number[idx] >= range[1]:
        return 0
    if range[0] >= range[1]:
        return 0
    return number[idx]

# print(findranege([11,4,23,9,10], [5,12]))
# print(findranege([1,3,2], [1,1]))
# print(findranege([7, 23, 1, 2], [2, 7]))

from sortedcontainers import SortedList
def obstacle(operations):
    intervals = SortedList()
    def add(x):
        toinsert = [x,x]
        idx = bisect_left(intervals, toinsert)
        remove = []
        if idx - 1 >= 0 and intervals[idx - 1][1] == x - 1:
            toinsert[0] = min(toinsert[0], intervals[idx - 1][0])
            remove.append(intervals[idx-1])
        if idx < len(intervals) and intervals[idx][0] == x + 1:
            toinsert[1] = max(toinsert[1], intervals[idx][1])
            remove.append(intervals[idx])
        for stale in remove: intervals.remove(stale)
        intervals.add(toinsert)
    def check(start, end):
        idx = bisect_left(intervals, [start, end])
        if idx - 1 >= 0 and intervals[idx - 1][1] >= start:
            return False
        if idx < len(intervals) and intervals[idx][0] <= end:
            return False
        return True
    res = []
    for op in operations:
        if op[0] == 1:
            add(op[1])
        else:
            _, x, size = op
            start, end = x - size, x - 1
            if check(start, end):
                res.append('1')
            else:
                res.append('0')
    print(intervals)
    return ''.join(res)

# print(obstacle([[1,2], [1,5], [1,3], [1,4], [1,6], [1,8], [2,5,2], [2,6,3], [2,2,1], [2,3,2], [2,5,2]]))




def candycrush2(board):
    n, m = len(board), len(board[0])
    crush = set()
    def check(x, y):
        nonlocal crush
        res = set()
        for dx, dy in ([0,1], [1,0], [0,-1], [-1,0]):
            if 0 <= dx + x < n and 0 <= dy + y < m:
                if board[dx + x][dy + y] == board[x][y]:
                    res.add((dx+x, dy+y))
        if len(res) >= 2:
            crush |= res
    for i in range(n):
        for j in range(m):
            check(i, j)
    for i, j in crush:
        board[i][j] = 0
    for col in range(m):
        wall = n - 1
        for row in range(n)[::-1]:
            if board[row][col] != 0:
                board[wall][col] = board[row][col]
                wall -= 1
        for i in range(wall + 1):
            board[i][col] = 0
    return board

# test = [[3,1,2,1], [1,1,1,4], [3,1,2,2], [3,1,2,4]]
# for row in test:
#     print(row)
# print('-------')
# res = candycrush2(test)
# for row in res:
#     print(row)
#

# find number of subset of an unordered list s.t. sum(subset) = target
def subsetSum(nums, S):
    dp = [0] * (S+1)  # dp[i] is the max number of ways to reach i
    dp[0] = 1
    for n in nums:
        for j in range(S + 1)[::-1]:
            if j - n >= 0:
                dp[j] += dp[j - n]  # the new way to get sum of j is to include n; and there're d[j - n] number of ways to do this # <- n_ways
    return dp[-1]


def count22(blacks, n, m):
    points = set()
    res = [0] * 5
    seen = set()
    for x, y in blacks:
        points.add((x, y))
    def valid(x, y):
        return 0 <= x < n and 0 <= y < m
    def helper(x, y):
        tmp = 0
        for dx, dy in [[0,0], [0,1], [1,0], [1,1]]:
            if not valid(x + dx, y + dy):
                return -1
            if (x + dx, y + dy) in points:
                tmp += 1
        res[tmp] += 1
        seen.add((x,y))
        return tmp



    for x, y in blacks:
        if (x,y) not in seen:
            helper(x, y)
        if valid(x-1, y) and (x-1, y) not in seen:
            helper(x-1, y)
        if valid(x-1, y-1) and (x-1, y-1) not in seen:
            helper(x-1, y-1)
        if valid(x, y-1) and (x, y-1) not in seen:
            helper(x, y-1)

    res[0] = (m-1) * (n-1) - sum(res)
    return res

# test = [[1, 1, 1], [1,1,1], [1,0,0]]

# blacks = []
# for i in range(n):
#     for j in range(m):
#         if test[i][j]:
#             blacks.append([i, j])
# print(count22(blacks, 3, 3))


def blackCount(blacks, rows, cols):
    blacks = set(map(tuple, blacks))
    res = [0] * 5
    for row_offset in range(rows-2+1):
        for col_offset in range(cols-2+1):
            tmp = set()
            for dx in (0, 1):
                for dy in (0, 1):
                    tmp.add((row_offset + dx, col_offset + dy))
            res[len(blacks & tmp)] += 1
    return res
# n = 101
# m = 201
# for _ in tqdm(range(10)):
#     test = []
#     for i in range(n):
#         for j in range(m):
#             if random.random() > 0.2:
#                 test.append([i, j])
#     random.shuffle(test)
#     a = blackCount(test, n, m)
#     b = count22(test, n, m)
#     assert a == b

def comb(nums, target):
    nums.sort()
    n = len(nums)
    used_ = [False] * n
    res = []
    def bt(used, tmp):
        if ''.join(tmp) == target:
            res.append(list(tmp))
        for i in range(n):
            # if i - 1 >= 0 and nums[i] == nums[i-1] and not used[i-1]:
            #     continue
            if not used[i]:
                used[i] = True
                bt(used, tmp + [nums[i]])
                used[i] = False
    bt(used_, [])
    return res
# print (comb(['21', '12'], '1221'))
from functools import lru_cache
def comb_word(nums, target: str):
    n = len(nums)
    m = len(target)
    res = 0

    @lru_cache(None)
    def helper(i, j):
        nonlocal res
        if j == m:
            res += 1
        # if i > n:
        #     return
        for ii in range(i, n):
            if target[j:].startswith(nums[ii]):
                helper(ii + 1, j + len(nums[ii]))

    helper(0, 0)
    return res
print (comb_word(['25', '2', '5', '12', '2', '21', '1'], '251221'))







def kduplicatesubarray(nums, k):
    n = len(nums)
    cnt = Counter()
    left = 0
    res = 0
    def valid(cnt):
        return sum([v >= 2 for v in cnt.values()]) >= k

    for i in range(n):
        cnt[nums[i]] += 1
        while valid(cnt):
            res += n - i
            cnt[nums[left]] -= 1
            left += 1
    return res

print(kduplicatesubarray([0,1,0,1,0], 2))





print(kduplicatesubarray([0,1,0,1,0], 2))
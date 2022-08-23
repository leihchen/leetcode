from typing import *
from collections import *
import math
from bisect import bisect_left
import random
from random import randint
from copy import deepcopy
from heapq import heappush, heappop, heapify
from itertools import combinations

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

# correct
def detectRect(points):
    counter = Counter()
    res = 0
    for x1, y1 in points:
        for (x3, y3), cnt in counter.items():
            if abs(x1-x3) == 0:
                continue
            res += cnt * counter[(x1, y3)] * counter[(x3, y1)]
        counter[(x1,y1)] += 1
    return res


# https://leetcode.com/problems/minimum-area-rectangle-ii/
# https://leetcode.com/problems/minimum-area-rectangle-ii/discuss/980956/Python3-center-point-O(N2)
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


print(detectSquare(test))
print(detectRect(test))
print(dectectRectUnAligned(test))
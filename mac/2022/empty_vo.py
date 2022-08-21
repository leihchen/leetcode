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
def closestMeetingNode(edges: List[int], node1: int, node2: int) -> int:
    # lca
    path1 = set()
    while node1 != -1:
        if node1 in path1:
            break
        path1.add(node1)
        node1 = edges[node1]
    path2 = set()
    print(path1)
    while node2 != -1:
        if node2 in path2:
            break
        path2.add(node2)
        if node2 in path1:
            return node2
        node2 = edges[node2]
    return -1
# print(closestMeetingNode([5,3,1,0,2,4,5],3,2,))


def reverseWord(a: str, delim='_'):
    s = a.split(delim)
    print(s)
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and s[l] == '':
            l += 1
        while l < r and s[r] == '':
            r -= 1
        s[l], s[r] = s[r], s[l]
        l += 1
        r -= 1
    return delim.join(s)

print(reverseWord('__hot_doge_'))


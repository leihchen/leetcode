from typing import *
from collections import *
import math
from bisect import bisect_left
import random
from random import randint
from copy import deepcopy
from heapq import heappush, heappop, heapify
from itertools import combinations

def findShortestSubArray(nums):
    cnt = Counter(nums)
    max_ = max(v for _, v in cnt.items())
    largest_index = {}
    n = len(nums)
    for i in range(n - 1, -1, -1):
        num = nums[i]
        if num not in largest_index:
            largest_index[num] = i
    res = n
    for i in range(n):
        if cnt[nums[i]] == max_:
            print(largest_index[nums[i]], i, max_)
            res = min(res, largest_index[nums[i]] - i + 1)
    return res
# print(findShortestSubArray([1,2,2,3,1]))


def singleNonDuplicate(nums: List[int]) -> int:
    # [1,1,2,3,3,4,4,8,8]
    # [0,1,2,3,4,5,6,7,8]
    # [1,1,0,0,0,0,0,0,0] nums[i] == nums[i+1] if i % 2 == 0
    #                     nums[i] == nums[i-1] else
    n = len(nums)
    left, right = 0, n-1
    while left <= right:
        mid = (left + right) // 2
        if mid == n - 1:
            break
        if (nums[mid] == nums[mid+1] and mid % 2 == 0) or (nums[mid] == nums[mid-1] and mid % 2 == 1):
            left = mid + 1
        else:
            right = mid - 1
    return nums[left]

# print(singleNonDuplicate([3,3,7,7,10,10,11]))


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isWord = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word):
        node = self.root
        for w in word:
            node = node.children[w]
        node.isWord = True

    def dfs(self, root, word):
        if not word:
            self.result |= root.isWord
            return
        if word[0] == '.':
            for k in root.children.keys():
                self.dfs(root.children[k], word[1:])

        elif word[0] in root.children:
            self.dfs(root.children[word[0]], word[1:])

    def search(self, word):
        self.result = False
        self.dfs(self.root, word)
        return self.result

def appealSum(s: str) -> int:
    seen = {}
    res = 0
    n = len(s)
    for i, c in enumerate(s):
        if c in seen:
            j = seen[c]
            print(c)
            print(j * (j + 1) // 2, (((i - j - 1) * (i - j) // 2) if i - j - 1 > 0 else 0), (n-1-i) * (n-i) // 2)
            adjust = j * (j + 1) // 2 + (((i - j - 1) * (i - j + 1) // 2) if i - j - 1 > 0 else 0) + (n-1-i) * (n-i) // 2
            res -= n * (n+1) // 2 - adjust
        seen[c] = i
        for s in range(1, i+2):
            print(min(s, len(seen)))
            res += min(s, i+1)
    print('uniq=', sum((i) * (n-i+1) for i in range(1, n+1)))
    return res

# print(appealSum('abbca'))

def findReplaceString(s: str, indices: List[int], sources: List[str], targets: List[str]) -> str:
    n = len(s)
    j = 0
    res = deque([])
    i = n
    replacement = [(indices[i], sources[i], targets[i]) for i in range(len(indices))]
    replacement.sort(reverse=True)

    while i >= 0:
        print(i, j)
        print(replacement[0])
        if j < len(replacement) and replacement[j][0] + len(replacement[j][1]) == i:
            src = replacement[j][1]
            if src == s[i - len(src):i]:
                res.appendleft(replacement[j][2])
                i = i - len(src)
                j += 1
                continue
            else:
                j += 1

        res.appendleft(s[i-1])
        i -= 1
    return ''.join(res)

# print(findReplaceString("abcd",
# [0, 2],
# ["a", "cd"],
# ["eee", "ffff"]))




# https://leetcode.com/discuss/interview-question/1621880/Google-or-Onsite-or-Maximum-total-or/1177349


def dijikstra(graph,s, n):
    visited = [False] * n
    distance = [float('inf')] * n
    distance[s] = 0
    pq = []
    heappush(pq, (distance[s], s))
    while pq:
        dist, node = heappop(pq)
        if visited[node]: continue
        for nei, w in graph[node]:
            if not visited[nei] and dist + w < distance[nei]:
                distance[nei] = dist + w
                heappush(pq, (distance[nei], nei))

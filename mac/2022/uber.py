from typing import *
from collections import *
import math
from bisect import bisect_left
import random

def findIternary(tickets):
    graph = defaultdict(deque)
    for src, dst in sorted(tickets):
        graph[src].append(dst)
    res = deque([])
    stack = ['JFK']
    while stack:
        while graph[stack[-1]]:
            stack.append(graph[stack[-1]].popleft())
        res.appendleft(stack.pop())
    return res
print(findIternary([["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]))

# 291. Word Pattern II
def wordPatternMatch(pattern: str, s: str) -> bool:
    # backtracking
    n, m = len(pattern), len(s)
    hashmap = {}
    stable = {}
    def bt(i, j):
        # print(hashmap)
        if i == n:
            return j == m
        if j >= m:
            return False
        result = False
        for end in range(j+1, m+1):
            # match s[j:end] with this pattern; put into hashmap
            added = False
            if s[j:end] in stable and stable[s[j:end]] != pattern[i]:
                continue
            if pattern[i] in hashmap and s[j:end] != hashmap[pattern[i]]:
                continue
            if pattern[i] not in hashmap and s[j:end] not in stable:
                hashmap[pattern[i]] = s[j:end]
                stable[s[j:end]] = pattern[i]
                added = True
            result |= bt(i+1, end)
            if added:
                del hashmap[pattern[i]]
                del stable[s[j:end]]
            if result:
                return True
        return result
    return bt(0, 0)

# 2187. Minimum Time to Complete Trips
def minimumTime(time: List[int], totalTrips: int) -> int:
    start, end = 1, math.ceil(totalTrips * min(time))
    while start <= end:
        mid = (start + end) // 2
        total = sum(mid // t for t in time)
        if total >= totalTrips:
            end = mid - 1
        else:
            start = mid + 1
    return start
# print(minimumTime([5,10,10], 9))

# 528. Random Pick with Weight
class Solution:
    def __init__(self, w: List[int]):
        self.prefix = []
        sum_ = 0
        for length in w:
            sum_ += length
            self.prefix.append(sum_)
    def pickIndex(self) -> int:
        pick = random.randint(1, self.prefix[-1])
        return bisect_left(self.prefix, pick)

# 986 Interval List Intersections
def intervalIntersection(firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    i = j = 0
    res = []
    while i < len(firstList) and j < len(secondList):
        head = max(firstList[i][0], secondList[j][0])
        tail = min(firstList[i][1], secondList[j][1])
        if head <= tail:
            res.append([head, tail])
        if secondList[j][1] < firstList[i][1]:
            j += 1
        else:
            i += 1
    return res

# firstList = [[0,2],[5,10],[13,23],[24,25]]
# secondList = [[1,5],[8,12],[15,24],[25,26]]
# print(intervalIntersection(firstList, secondList))

# 399. Evaluate Division
def calcEquation(equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
    # bfs
    graph = defaultdict(list)
    n = len(equations)
    for i in range(n):
        src, dst = equations[i]
        graph[src].append((dst, values[i]))
        graph[dst].append((src, 1 / values[i]))
    def query(src, dst):
        if dst not in graph: return -1
        q = deque([(src, 1)])
        visited = {src}
        while q:
            node, weight = q.popleft()
            if node == dst:
                return weight
            for nei, edge_weight in graph[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, weight * edge_weight))
        return -1
    return [query(src, dst) for src, dst in queries]

# 79. Word Search
def exist(board: List[List[str]], word: str) -> bool:
    dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
    n, m = len(board), len(board[0])

    def dfs(i, j, idx):
        if idx == len(word):
            return True
        res = False
        if 0 <= i < n and 0 <= j < m and board[i][j] == word[idx]:
            tmp = board[i][j]
            board[i][j] = '#'
            for dir in dirs:
                res |= dfs(i + dir[0], j + dir[1], idx + 1)
            board[i][j] = tmp
        return res

    for i in range(n):
        for j in range(m):
            if dfs(i, j, 0):
                return True
    return False


class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        self.mask = 0
        self.charReversed = characters
        self.n = len(characters)
        self.k = combinationLength

    def next(self) -> str:
        self.mask += 1
        while self.mask < 2 ** self.n - 1 and bin(self.mask).count('1') != self.k:
            self.mask += 1
        result = [self.charReversed[i] for i in range(self.n) if self.mask & (1 << i)]
        return ''.join(result)

    def hasNext(self) -> bool:
        return self.mask < 2 ** self.n - 1

# ci = CombinationIterator('abc', 2)
# print(ci.next())
# print(ci.next())
# print(ci.next())

def maxKilledEnemies(grid):
    m, n = len(grid), len(grid[0])
    nums = [[0] * n for _ in range(m)]
    for i in range(m):
        row_hits = 0
        for j in range(n):
            if grid[i][j] == 'E':
                row_hits += 1
            elif grid[i][j] == 'W':
                row_hits = 0
            else:
                nums[i][j] = row_hits
    for i in range(m):
        row_hits = 0
        for j in range(n)[::-1]:
            if grid[i][j] == 'E':
                row_hits += 1
            elif grid[i][j] == 'W':
                row_hits = 0
            else:
                nums[i][j] += row_hits

    for j in range(n):
        col_hits = 0
        for i in range(m):
            if grid[i][j] == 'E':
                col_hits += 1
            elif grid[i][j] == 'W':
                col_hits = 0
            else:
                nums[i][j] += col_hits
    res = 0
    for j in range(n):
        col_hits = 0
        for i in range(m)[::-1]:
            if grid[i][j] == 'E':
                col_hits += 1
            elif grid[i][j] == 'W':
                col_hits = 0
            else:
                nums[i][j] += col_hits
                res = max(nums[i][j], res)
    return res

# print(maxKilledEnemies([["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]))

# 815. Bus Routes
def numBusesToDestination(routes: List[List[int]], source: int, target: int):
    stop2bus = defaultdict(set)
    for i, route in enumerate(routes):
        for j in route:
            stop2bus[j].add(i)
    q = deque([(source, 0)])
    seen = {source}
    while q:
        stop, hop = q.popleft()
        if stop == target: return hop
        for next_bus in stop2bus[stop]:
            for next_stop in routes[next_bus]:
                if next_stop not in seen:
                    q.append([next_stop, hop + 1])
                    seen.add(next_stop)
            routes[next_bus] = []
    return -1

print(numBusesToDestination([[1,2,7],[3,6,7]], 1, 6))


def nearestPalindromic(S):
    K = len(S)
    candidates = [str(10 ** k + d) for k in (K - 1, K) for d in (-1, 1)]
    prefix = S[:(K + 1) // 2]
    P = int(prefix)
    for start in map(str, (P - 1, P, P + 1)):
        candidates.append(start + (start[:-1] if K % 2 else start)[::-1])
    def delta(x):
        return abs(int(S) - int(x))

    ans = None
    for cand in candidates:
        if cand != S and not cand.startswith('00'):
            if (ans is None or delta(cand) < delta(ans) or
                    delta(cand) == delta(ans) and int(cand) < int(ans)):
                ans = cand
    return ans
nearestPalindromic('245')









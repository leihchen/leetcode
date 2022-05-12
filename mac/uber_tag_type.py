from typing import *
from collections import *
import math
from bisect import bisect_left
import random
from random import randint
from copy import deepcopy
from heapq import heappush, heappop, heapify
from itertools import combinations


## TODO: BFS
# 815. Bus Routes
def numBusesToDestination(routes: List[List[int]], source: int, target: int):
    stop2bus = defaultdict(set)
    for i, route in enumerate(routes):
        for j in route:
            stop2bus[j].add(i)  # bus you can take at a stop
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
            routes[next_bus] = []  # remove the bus
    return -1

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

# 773. Sliding Puzzle
def slidingPuzzle(board):
    moves, used, cnt = {0: {1, 3}, 1:{0, 2, 4}, 2:{1, 5}, 3:{0, 4}, 4:{1, 3, 5}, 5:{2, 4}}, set(), 0
    s = "".join(str(c) for row in board for c in row)
    q = [(s, s.index("0"))]
    while q:
        new = []
        for s, i in q:
            used.add(s)
            if s == "123450":
                return cnt
            arr = [c for c in s]
            for move in moves[i]:
                new_arr = arr[:]
                new_arr[i], new_arr[move] = new_arr[move], new_arr[i]
                new_s = "".join(new_arr)
                if new_s not in used:
                    new.append((new_s, move))
        cnt += 1
        q = new
    return -1

## TODO: DFS
# 212. Word Search II
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isWord = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for w in word:
            node = node.children[w]
        node.isWord = True

    def search(self, word: str) -> bool:
        node = self.root
        for w in word:
            node = node.children.get(w)
            if node == None:
                return False
        return node.isWord

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for w in prefix:
            node = node.children.get(w)
            if node == None:
                return False
        return True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        n, m = len(board), len(board[0])
        visited, res = set(), set()
        trie = Trie()
        for word in words:
            trie.insert(word)

        def dfs(i, j, string):
            if i < 0 or i >= n or j < 0 or j >= m: return
            if (i, j) in visited:  return
            string += board[i][j]
            if not trie.startsWith(string): return
            if trie.search(string): res.add(string)
            visited.add((i, j))
            dfs(i + 1, j, string)
            dfs(i - 1, j, string)
            dfs(i, j + 1, string)
            dfs(i, j - 1, string)
            visited.remove((i, j))

        for x in range(n):
            for y in range(m):
                dfs(x, y, '')
        return res

# 529. Minesweeper
def updateBoard(board: List[List[str]], click: List[int]) -> List[List[str]]:
    n, m = len(board), len(board[0])

    def dfs(i, j):
        if board[i][j] != 'E':
            return
        directions = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
        mine_count = 0
        for d in directions:
            ni, nj = i + d[0], j + d[1]
            if 0 <= ni < n and 0 <= nj < m and board[ni][nj] == 'M':
                mine_count += 1
        if mine_count == 0:
            board[i][j] = 'B'
        else:
            board[i][j] = str(mine_count)
            return
        for d in directions:
            ni, nj = i + d[0], j + d[1]
            if 0 <= ni < n and 0 <= nj < m:
                dfs(ni, nj)

    i, j = click
    if board[i][j] == 'M':
        board[i][j] = 'X'
        return board
    dfs(i, j)
    return board

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
            result |= bt(i+1, end)  # match pair exists and success
            if added:
                del hashmap[pattern[i]]
                del stable[s[j:end]]
            if result:
                return True
        return result
    return bt(0, 0)

# 17. Letter Combinations of a Phone Number
def letterCombinations(digits: str) -> List[str]:
    if not digits: return []
    n2l = {2: 'abc', 3: 'def', 4: 'ghi', 5: 'jkl', 6: 'mno', 7: 'pqrs', 8: 'tuv', 9:'wxyz'}
    n = len(digits)
    res = []
    def bt(i, tmp):
        if len(tmp) == n:
            res.append(''.join(tmp))
            return
        for j in n2l[int(digits[i])]:
            tmp.append(j)
            bt(i+1, tmp)
            tmp.pop()
    bt(0, [])
    return res

## TODO: monotonic stack
# 	1438 Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
def longestSubarray(nums: List[int], limit: int) -> int:
    minheap, maxheap = [], []
    # res = left = 0
    # for i, v in enumerate(nums):
    #     heappush(minheap, (v, i))
    #     heappush(maxheap, (-v, i))
    #     while minheap and maxheap and -maxheap[0][0] - minheap[0][0] > limit:
    #         left = min(maxheap[0][1], minheap[0][1]) + 1
    #         while minheap[0][1] < left:
    #             heappop(minheap)
    #         while maxheap[0][1] < left:
    #             heappop(maxheap)
    #     res = max(res, i - left + 1)
    # return res
    ####
    asc, desc = deque([]), deque([])
    left, res = 0, float('-inf')
    for i in range(len(nums)):
        while asc and asc[-1] > nums[i]: asc.pop()
        while desc and desc[-1] < nums[i]: desc.pop()
        asc.append(nums[i])
        desc.append(nums[i])
        if desc[0] - asc[0] > limit:
            if asc[0] == nums[left]: asc.popleft()
            if desc[0] == nums[left]: desc.popleft()
            left += 1
        res = max(res, i - left + 1)
    return res

# 1950. Maximum of Minimum Values in All Subarrays
def findMaximums(nums: List[int]) -> List[int]:
    cur_nums = deepcopy(nums)
    n = len(nums)
    ans = [0] * n
    for i in range(n):                    # for each window size
        cur_ans = 0
        for j in range(n-i):              # compare and find the maximum among all minimum values
            cur = cur_nums[j]
            cur = min(cur, nums[j + i])   # get the minimum of current window size at index `j`
            cur_nums[j] = cur
            cur_ans = max(cur_ans, cur)   # compare to get the maximum
        ans[i] = cur_ans
    return ans
# The second value from the top of the stack is the first value less than the value at the top of stack, reading from right to left
# current_value = arr[i], current_idx = i
# stack_top_value = stack[-1], stack_top_idx = index of stack[-1] in arr
# second_stack_top_value = stack[-2], second_stack_top_idx = index of stack[-2] in arr
#  0 1 2  3
# [1,3,2, 1]
#    ---
# index =  0 2
# stack = [1,2]; 1
# 2 is the min values bewteen 0 + 1 and 3 - 1
def findMaximums(nums: List[int]) -> List[int]:
    n = len(nums)
    dp = [0] * n
    stack = []
    for i in range(n):
        while stack and nums[stack[-1]] >= nums[i]:
            min_idx = stack.pop()
            left = stack[-1] + 1 if stack else 0
            window = i - left - 1
            dp[window] = max(dp[window], nums[min_idx])
        stack.append(i)
    else:
        while stack:
            min_idx = stack.pop()
            left = stack[-1] + 1 if stack else 0
            window = n - left - 1
            dp[window] = max(dp[window], nums[min_idx])

    for i in range(n-2, -1, -1):
        dp[i] = max(dp[i], dp[i+1])
    return dp

## TODO: heap
# 692. Top K Frequent Words
class ReversedLex:
    def __init__(self, word):
        self.word = word

    def __lt__(self, other):
        return self.word > other.word

class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        # cnt = [(-v, k) for k, v in Counter(words).items()]
        # res = heapq.nsmallest(k, cnt)
        # return [k for _, k in res]
        cnt = Counter(words)
        minheap = []
        for word, v in cnt.items():
            heappush(minheap, [v, ReversedLex(word)])
            if len(minheap) > k:
                heappop(minheap)
        res = []
        while minheap:
            res.append(heappop(minheap)[1].word)
        return res[::-1]


## TODO: Palindrome
# 1400. Construct K Palindrome Strings
def canConstruct(s: str, k: int) -> bool:
    cnt = Counter(s)
    if len(s) < k:
        return False
    return sum([1 if v % 2 == 1 else 0 for v in cnt.values()]) <= k

def nearestPalindromic(S: str) -> str:
    K = len(S)
    candidates = [str(10 ** k + d) for k in (K - 1, K) for d in (-1, 1)]
    print(candidates)
    prefix = S[:(K + 1) // 2]
    P = int(prefix)
    for start in map(str, (P - 1, P, P + 1)):
        candidates.append(start + (start[:-1] if K % 2 else start)[::-1])
    print(candidates)
    def delta(x):
        return abs(int(S) - int(x))

    ans = None
    for cand in candidates:
        if cand != S and not cand.startswith('00'):
            if (ans is None or delta(cand) < delta(ans) or
                    delta(cand) == delta(ans) and int(cand) < int(ans)):
                ans = cand
    return ans


# 1842. Next Palindrome Using Same Digits
def nextPermutation(sList: List[int]):
    i = len(sList) - 2
    while i >= 0 and sList[i] >= sList[i + 1]:
        i -= 1
    if i < 0:
        return []

    j = len(sList) - 1
    while j >= 0 and sList[i] >= sList[j]:
        j -= 1

    sList[i], sList[j] = sList[j], sList[i]
    sList[i + 1:] = reversed(sList[i + 1:])

def nextPalindrome(num: str) -> str:
    num_list = list(num)
    mid = len(num) // 2
    midStr = "" if (len(num) % 2 == 0) else num_list[mid]

    left_greater = nextPermutation(num_list[: mid])
    if not left_greater:
        return ""

    return "".join(left_greater) + midStr + "".join(reversed(left_greater))

# 305. Number of Islands II
class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.count = 0

    def find(self, x):
        if x != self.p[x]:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return
        self.p[px] = py
        self.count -= 1


class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        dsu = DSU(n * m)
        board = [[0] * n for _ in range(m)]
        res = []
        for x, y in positions:
            if board[x][y] == 1:
                res.append(dsu.count)
                continue
            board[x][y] = 1
            dsu.count += 1
            for di in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                xx, yy = x + di[0], y + di[1]
                if 0 <= xx < m and 0 <= yy < n and board[xx][yy] == 1:
                    dsu.union(x * n + y, xx * n + yy)
            res.append(dsu.count)
        return res

# 945. Minimum Increment to Make Array Unique
def minIncrementForUnique(nums: List[int]) -> int:
    # res = need = 0
    # for i in sorted(nums):
    #     res += max(need - i, 0)
    #     need = max(need + 1, i + 1)
    # return res

    root = {}

    def find(x):
        if x not in root:
            root[x] = x
        else:
            root[x] = find(root[x] + 1)
        return root[x]

    return sum(find(a) - a for a in nums)

## TODO: binary search
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

## TODO: sweep line
# 986. Interval List Intersections
def intervalIntersection(firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    m, n = len(firstList), len(secondList)
    i, j = 0, 0
    res = []
    while i < m and j < n:
        first = firstList[i]
        second = secondList[j]
        if first[1] < second[0]:
            i += 1
            continue
        if first[0] > second[1]:
            j += 1
            continue
        res.append([max(first[0], second[0]), min(first[1], second[1])])
        if first[1] > second[1]:
            j += 1
        else:
            i += 1
    return res
# 56. Merge Intervals
def merge(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    res = []
    start, end = intervals[0][0], intervals[0][1]
    for i in range(1, len(intervals)):
        if start <= intervals[i][0] and end >= intervals[i][0]:
            end = max(intervals[i][1], end)
        else:
            res.append([start, end])
            start, end = intervals[i][0], intervals[i][1]

    res.append([start, end])
    return res
## TODO: DP
# 361. Bomb Enemy
def maxKilledEnemies(grid: List[List[str]]) -> int:
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

# 174 dungeon game
def calculateMinimumHP(dungeon):
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
    dp[m - 1][n], dp[m][n - 1] = 1, 1
    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            dp[i][j] = max(min(dp[i+1][j] - dungeon[i][j], dp[i][j+1] - dungeon[i][j]), 1)
    return dp[0][0]

## TODO: tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def kthSmallest(root: Optional[TreeNode], k: int) -> int:
    stack = []
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0: return root.val
        root = root.right
    return -1
    ###
    res = -1
    def inorder(root):
        nonlocal res, k
        if not root: return
        inorder(root.left)
        k -= 1
        if k == 0:
            res = root.val
            return
        inorder(root.right)
    inorder(root)
    return res

# 428. Serialize and Deserialize N-ary Tree
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
class Codec:
    def serialize(self, root: 'Node') -> str:
        """Encodes a tree to a single string.

        :type root: Node
        :rtype: str
        """
        res = []

        def preorder(node):
            if not node: return
            res.append(str(node.val))
            for child in node.children:
                preorder(child)
            res.append('#')

        preorder(root)
        return ','.join(res)

    def deserialize(self, data: str) -> 'Node':
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: Node
        """
        if not data: return None
        tokens = deque(data.split(','))
        root = Node(int(tokens.popleft()), [])

        def helper(node):
            if not tokens:
                return
            while tokens[0] != '#':
                value = tokens.popleft()
                child = Node(int(value), [])
                node.children.append(child)
                helper(child)
            tokens.popleft()  # remove '#'

        helper(root)
        return root

# 427. Construct Quad Tree
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight

def construct(grid: List[List[int]]) -> 'Node':
    def allSame(grid):
        for row in grid:
            for elem in row:
                if grid[0][0] != elem:
                    return False
        return True

    if allSame(grid):
        return Node(grid[0][0], True, None, None, None, None)
    n = len(grid)
    topLeft = [row[:n // 2] for row in grid[: n // 2]]
    topRight = [row[n // 2:] for row in grid[: n // 2]]
    bottomLeft = [row[:n // 2] for row in grid[n // 2:]]
    bottomRight = [row[n // 2:] for row in grid[n // 2:]]
    return Node(grid[0][0], False, construct(topLeft), construct(topRight), construct(bottomLeft),
                construct(bottomRight))

## TODO: graph
# 332. Reconstruct Itinerary
def findItinerary(tickets: List[List[str]]) -> List[str]:
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

## TODO: misc.
# 465 Optimal Account Balancing
def minTransfers(transactions):
    """
    Get all persons that have outstanding balance + or -, and discard others
    Greedily find smallest group of persons whose balances sum to zero, and set them to zero.
    Do this by getting all combinations of size 2, then size 3 etc. The moment any size group sums to zero,
    recurse on remaining outstanding persons. Because a greedy solution is guaranteed to exist and is optimal
    e.g.
    6->0 : 50, 1->6 : 40, 2->6 : 10, 6->3 : 40, 6->4 : 40, 5->6 : 25, 6->7 : 10, 7->1 : 10
    balance : 0:50, 1:-30, 2:-10, 3:40, 4:10, 5:-25, 6:-35, 7:0
    remove 7 since it's already at 0, and participating in any give/take only increases transaction size
    Then try to find any combination of size 2 that can cancel each other's debts
    e.g. 2 & 4. 4 gives 10 to 2, 1 transaction. Then 2 & 4's balances are set to 0, then remove them.
    Recurse on remaining.
    Again try to find any combinations of size 2. None. So find combinations of size 3, size 4, size 5.
    size 5 - all remaining add up to 90 - 90. 4 transactions
    Total 5 transactions
    """

    def outstanding(balance):  # Get all persons with either + or - money
        return {person: money for person, money in balance.items() if money != 0}

    def min_trans(balance):
        if not balance:
            return 0
        res = float('inf')
        for size in range(2, len(balance) + 1):  # Greedily start from smallest size of 2
            for group in combinations(balance.keys(), size):  # Get all combinations
                if sum(balance[person] for person in group) == 0:  # If they can cancel each others debts
                    remaining_balances = {k: v for k, v in balance.items() if k not in group}
                    transactions = size - 1  # Number of transactions needed is size-1
                    res = min(res, transactions + min_trans(
                        outstanding(remaining_balances)))  # Recurse on remaining outstanding balances
        return res

    balance = defaultdict(int)
    for u, v, money in transactions:  # Get final balance of each person
        balance[u] -= money
        balance[v] += money

    return min_trans(outstanding(balance))

# 384. Shuffle an Array
class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.copy = deepcopy(nums)

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        self.nums = deepcopy(self.copy)
        return self.nums

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        n = len(self.nums)
        for i in range(n):
            j = randint(i, n - 1)
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
        return self.nums

# 528. Random Pick with Weight
class Solution:

    def __init__(self, w: List[int]):
        self.prefix = []
        sum_ = 0
        for ww in w:
            sum_ += ww
            self.prefix.append(sum_)

    def pickIndex(self) -> int:
        # [1,4]  [1,2,3,4]
        return bisect_left(self.prefix, random.randint(1, self.prefix[-1]))

# 1861. Rotating the Box
def rotateTheBox(box: List[List[str]]) -> List[List[str]]:
    STONE = '#'
    OBS = '*'
    EMPTY = '.'
    m = len(box[0])
    for row in box:
        empty = m - 1
        for i in range(m)[::-1]:
            if row[i] == STONE:
                row[empty], row[i] = row[i], row[empty]
                empty -= 1
            if row[i] == OBS:
                empty = i - 1

    return zip(*box[::-1])

# 1286. Iterator for Combination
class CombinationIterator:
    def __init__(self, characters: str, combinationLength: int):
        self.charReversed = characters
        self.n = n = len(characters)
        self.k = k = combinationLength
        self.mask = (1 << n) - (1 << n - k)

    def next(self) -> str:
        result = [self.charReversed[i] for i in range(self.n) if self.mask & 1 << (self.n - i - 1)]
        self.mask -= 1
        while self.mask < 2 ** self.n and bin(self.mask).count('1') != self.k:
            self.mask -= 1

        return ''.join(result)

    def hasNext(self) -> bool:
        return self.mask > 0

# 68. Text Justification
def fullJustify(words: List[str], maxWidth: int) -> List[str]:
    curLen, cur = 0, []
    res = []
    for i in range(len(words)):
        word = words[i]
        if len(word) + len(cur) + curLen > maxWidth:
            # flush
            if len(cur) == 1:
                res.append(cur[0] + ' ' * (maxWidth - curLen))
            else:
                for i in range(maxWidth - curLen):
                    cur[i % (len(cur) - 1)] += ' '
                res.append(''.join(cur))
            curLen, cur = 0, []
        curLen += len(word)
        cur.append(word)
    return res + [' '.join(cur) + ' ' * (maxWidth - curLen - len(cur) + 1)]

# 247. Strobogrammatic Number II
def findStrobogrammatic(n: int) -> List[str]:
    even = ['11', '69', '88', '96', '00']
    odd = ['0', '1', '8']
    if n == 1: return odd
    if n == 2: return even[:-1]
    if n % 2:
        middle = odd
        pre = findStrobogrammatic(n-1)
    else:
        middle = even
        pre = findStrobogrammatic(n-2)
    premid = (n-1) // 2
    return [p[:premid] + c + p[premid:] for c in middle for p in pre]

# 224. Basic Calculator
class Solution:
    i = 0
    def calculate(self, s: str) -> int:
        s = s.replace(' ', '')
        sign = '+'
        num = 0
        stack = []
        if not s: return 0
        while self.i < len(s):
            c = s[self.i]
            self.i += 1
            if c.isdigit():
                num = 10 * num + int(c)
            if c == '(':  # IMPORTANT check ( before flushing into stack
                num = self.calculate(s)
            if not c.isdigit() or self.i >= len(s):  # any non number char marks the end of the num
                if sign == '+':
                    stack.append(num)
                if sign == '-':
                    stack.append(-num)
                if sign == '*':
                    stack.append(stack.pop() * num)
                if sign == '/':
                    stack.append(int(stack.pop() / num))
                num = 0
                sign = c

            if c == ')':
                break
        return sum(stack)

# 1428. Leftmost Column with at Least a One
def leftMostColumnWithOne(binaryMatrix: 'BinaryMatrix') -> int:
    n, m = binaryMatrix.dimensions()
    row, col = 0, m - 1
    res = -1
    while row < n and col >= 0:
        if binaryMatrix.get(row, col) == 1:
            res = col
            col -= 1
        else:
            row += 1
    return res

# 380. Insert Delete GetRandom O(1)
class RandomizedSet:

    def __init__(self):
        self.array = []
        self.hashmap = {}

    def insert(self, val: int) -> bool:
        if val in self.hashmap:
            return False
        self.hashmap[val] = len(self.array)
        self.array.append(val)

        return True

    def remove(self, val: int) -> bool:
        if val not in self.hashmap:
            return False
        last_val = self.array[-1]
        self.hashmap[last_val] = self.hashmap[val]
        self.array[self.hashmap[val]], self.array[-1] = self.array[-1], self.array[self.hashmap[val]]
        self.array.pop()
        del self.hashmap[val]
        return True

    def getRandom(self) -> int:
        return self.array[random.randint(0, len(self.array) - 1)]



# phone screening
# import requests
# import mysql.connector
# import pandas as pd

# In a company, has different levels,

# Given info:

# (Alan James) means Alan reports to James

# (Alex, Tiffany)

# (Ray, Tiffany)

# (James, Daniel)

# (Tiffany, Daniel)

# Find the least common leader for given two people.

# Exp 1:

# Alex, Ray -> Tiffany

# Exp 2:

# Tiffany, James -> Daniel

# Exp 3:

# Ray, James -> Daniel
# Ray -> Tiffany -> Daniel
# James -> Daniel

# */

# Alex -> Tiffany -> Daniel
# Ray -> Tiffany -> Daniel

# Tiffany, Alex
# V vertice, E edges
# O(V) Memory
# O(V + E)
print('Hello')


def leastCommonLeader(edgeList, person1, person2):
    # people -> people2 -> people3 -> ... ->
    # people -> people2 -> people3 -> ... ->
    # dict<people, leader>:
    leader = {}  # leader[people] = leader
    for person, l in edgeList:
        leader[person] = l
    # find report chain of person1
    leader_person1 = set([person1])  # leader set()
    while person1 in leader:
        leader_person1.add(leader[person1])
        person1 = leader[person1]
    # find the chain of person2, check if leader of person2 in the leader_person1 O(1)
    print(leader_person1)
    # Ray -> Tiffany -> Daniel
    if person2 in leader_person1: return person2
    # James -> Daniel
    while person2 in leader:
        if leader[person2] in leader_person1:
            return leader[person2]  # <-
        person2 = leader[person2]
    return '#'


# edges = [('Alan', 'James'), ('Alex', 'Tiffany'), ('Ray', 'Tiffany') ,('James', 'Daniel'), ('Tiffany', 'Daniel')]
# print(leastCommonLeader(edges, 'Alex', 'Tiffany'))  # Alex -> Tiffany -> Daniel
# print(leastCommonLeader(edges, 'Tiffany', 'Alex'))
# print(leastCommonLeader(edges, 'Ray', 'James'))
# print(leastCommonLeader(edges, 'James', 'Ray'))      # Ray -> Tiffany -> Daniel      # James -> Daniel
# print(leastCommonLeader(edges, 'Ray', 'Any'))
# DAG, tree no cycle


# given 2d array, represent 1 and 0, 1 means a path, 0 means a wall
# if there is a path top-left to bottom-right
# [0,0] -> [m-1,n-1]
dris = [[0, 1], [1, 0], [-1, 0], [0, -1]]


# dfs, bfs travese the matrix
# dfs
# dfs with backtracking
# mark matrix[i][j] = -1
#  call dfs() -> neighour
# unmark matrix[i][j]

def find_path(matrix):
    m, n = len(matrix), len(matrix[0])

    def dfs(i, j):
        result = False
        if 0 <= i < m and 0 <= j < n and matrix[i][j] == 1:
            if i == m - 1 and j == n - 1:
                return True
            # temp = matrix[i][j]
            matrix[i][j] = 0
            for dx, dy in dris:
                x, y = i + dx, j + dy
                if dfs(x, y):
                    result = True
                    break
            matrix[i][j] = 1
        return result

    return dfs(0, 0)


board = [[1, 1, 1],
         [0, 1, 1],
         [0, 0, 1]]

print(find_path(board))
# O(MN)
#

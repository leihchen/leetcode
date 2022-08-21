from collections import defaultdict, OrderedDict
from itertools import groupby
from math import ceil

# leetcode 696
from collections import deque, Counter, OrderedDict
## https://www.1point3acres.com/bbs/thread-804015-1-1.html
# 1048 Longest String Chain
class Solution:
    def longestStrChain(self, words) -> int:
        dp = {}
        for w in sorted(words, key=len):
            dp[w] = max(dp.get(w[:i] + w[i + 1:], 0) + 1 for i in range(len(w)))
        return max(dp.values())

# 2021.07 secret array
def analogous_secret_array(consecutiveDifference, lower, upper):  # conute analogous arrays
    prefix = [0]
    max_, min_ = 0, 0
    for i in consecutiveDifference:
        prefix.append(prefix[-1] - i)
        max_ = max(max_, prefix[-1])
        min_ = min(min_, prefix[-1])
    # print(max_, min_)
    return max(0, 1 + (upper-lower) - (max_ - min_))
# print(analogous_secret_array([-2, -1, -2, 5], 3, 10))

## https://www.1point3acres.com/bbs/thread-804270-1-1.html
# Aldin Array
# not unique, choose the smallest index possible
# extending starting point, whenever we find out a point unreachable
# choose it as the starting point
def optimalPoint(magic, dist):
    if sum(magic) < sum(dist): return -1
    res = 0
    remaining = 0
    for i in range(len(magic)):
        if remaining < 0:
            remaining = 0
            res = i
        remaining += magic[i] - dist[i]
    return res

# 134. Gas Station, guaranteed to be unique
# got the gas stations hardest to get (lowest tank)
def gasStop(gas, cost):
    if sum(gas) < sum(cost): return -1
    tank = 0
    min_, idx = 0, -1
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0 and tank < min_:
            idx = i
            min_ = tank
    return (idx + 1) % len(gas)
# print(optimalPoint([2, 4, 5, 2], [4, 3, 1, 3]))
# print(optimalPoint([8, 4, 1, 9], [10, 9, 3, 5]))



## https://www.1point3acres.com/bbs/tag-4142-3.html
## https://www.1point3acres.com/bbs/thread-815316-1-1.html
## https://www.1point3acres.com/bbs/thread-814853-1-1.html
## https://www.1point3acres.com/bbs/thread-804444-1-1.html



# 1654 Minimum Jumps to Reach Home
class Solution:
    def minimumJumps(self, forbidden, a: int, b: int, x: int) -> int:
        max_val = max([x] + forbidden) + a + b  # see proof in discuss
        jumps = [0] + [float('inf')] * max_val
        for pos in forbidden: jumps[pos] = -1
        q = deque([0])
        while q:
            pos = q.popleft()
            if pos + a <= max_val and jumps[pos + a] > jumps[pos] + 1:
                q.append(pos+a)
                jumps[pos+a] = jumps[pos] + 1
            if pos - b >= 0 and jumps[pos-b] > jumps[pos] + 1:
                jumps[pos-b] = jumps[pos] + 1
                if pos - b + a <= max_val and jumps[pos-b+a] > jumps[pos] + 2:
                    q.append(pos-b+a)
                    jumps[pos-b+a] = jumps[pos] + 2
        return jumps[x] if jumps[x] != float('inf') else -1

# 696 Count Binary Substrings
# 不用sliding windows 只有01情况下 对于每个交接点计算可以生成的valid substr数量
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        subcnt = [len(list(g)) for _, g in groupby(s)]
        res = 0
        for i in range(1, len(subcnt)):
            res += min(subcnt[i], subcnt[i-1])
        return res
# 104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

def anagram_modify(s1, s2):
    if len(s1) != len(s2): return -1
    cnt = Counter()
    for i in s1:
        cnt[i] += 1
    for i in s2:
        cnt[i] -= 1
    return sum(abs(v) for v in cnt.values()) // 2
# print(anagram_modify('ddcf', 'cedk'))

def recover_2d_prefixsum(prefix):
    # element[i][j] = prefix[i][j] - prefix[i-1][j] - prefix[i][j-1] + prefix[i-1][j-1]
    n, m = len(prefix), len(prefix[0])
    nums = [[0 for _ in range(m)] for _ in range(n)]
    nums[0][0] = prefix[0][0]
    for i in range(n):
        for j in range(m):
            nums[i][j] = prefix[i][j] - (prefix[i-1][j] if i > 0 else 0) - (prefix[i][j-1] if j > 0 else 0) + (prefix[i-1][j-1] if i > 0 and j > 0 else 0)
    return nums
# def recover_2d_prefixsum(prefix):
#     n, m = len(prefix), len(prefix[0])
#     for s in range(1, m+n-1)[::-1]:
#         for i in range(n):
#             j = s - i
#             if 0 <= j < m:
#                 prefix[i][j] = prefix[i][j] - (prefix[i - 1][j] if i > 0 else 0) - (prefix[i][j - 1] if j > 0 else 0) + (
#                     prefix[i - 1][j - 1] if i > 0 and j > 0 else 0)
#     print(prefix)
# print(recover_2d_prefixsum([[10,30,60, 100], [15,45, 95, 95+25+40], [17, 51, 107, 107+9+25+40]]))


# reductor array
def reductor_array(a, b, d):
    res = 0
    bmin, bmax = min(b), max(b)
    for aa in a:
        if abs(aa - bmin) > d or abs(aa - bmax) > d:
            res += 1
    return res

# 1507. Reformat Date
class Solution:
    def reformatDate(self, date: str) -> str:
        m = {"Jan": '01', "Feb": '02', "Mar": '03', "Apr": '04', "May": '05', "Jun": '06', "Jul": '07', "Aug": '08', "Sep": '09', "Oct": '10', "Nov": '11', "Dec": '12'}
        dmy = date.split(' ')
        d = dmy[0][:-2]
        if len(d) == 1: d = '0' + d
        return '-'.join([dmy[2], m[dmy[1]], d])

def costum_sort(nums):
    n = len(nums)
    left, right = 0, n-1
    res = 0
    while left < right:
        if nums[left] % 2 == 1:
            while nums[right] % 2 == 1 and left < right:
                right -= 1
            if left >= right:
                break
            res += 1
        left += 1
    return res
# print(costum_sort([8, 5, 11, 4, 6]))


class TreeNode:
    def __init__(self, val='', left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class DSU:
    def __init__(self, n):
        self.p = [i for i in range(n)]
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self,x , y):
        self.p[self.find(x)] = self.find(y)

class SolutionTree:
    def __init__(self):
        self.tree = dict()

    def connectNode(self, p, c):
        pnode = self.tree.get(p, TreeNode(p))
        cnode = self.tree.get(c, TreeNode(c))
        if not pnode.left:
            pnode.left = cnode
        else:
            if pnode.left.val < cnode.val:
                pnode.right = cnode
            else:
                pnode.right = pnode.left
                pnode.left = cnode
        self.tree[p] = pnode
        self.tree[c] = cnode

    def preorderPrint(self, root):
        if not root: return ''
        return '(' + root.val + self.preorderPrint(root.left) + self.preorderPrint(root.right) + ')'

    def sexp(self, pairs):
        n = len(pairs)
        if n == 0: return ''
        adjMat = [[0 for _ in range(26)] for _ in range(26)]
        indegree = [0] * 26
        outdegree = [0] * 26
        dsu = DSU(26)
        for parent, child in pairs:
            p = ord(parent) - ord('A')
            c = ord(child) - ord('A')
            if outdegree[p] == 2 and adjMat[p][c] != 1:
                return 'E1'  # more than 2 children for a node
            elif adjMat[p][c] == 1:
                return 'E2'  # duplicate edges
            elif dsu.find(p) == dsu.find(c):
                return 'E3'  # loop
            adjMat[p][c] = 1
            outdegree[p] += 1
            indegree[c] += 1
            dsu.union(p, c)
            self.connectNode(parent, child)
        # check for root
        rootcnt = 0
        root = None
        for x in self.tree.keys():
            if indegree[ord(x) - ord('A')] == 0:
                rootcnt += 1
                if rootcnt > 1:
                    return 'E4'
                root = self.tree[x]
        if not root: return 'E5'
        return self.preorderPrint(root)

testE1 = [['A', 'B'], ['A', 'C'], ['B', 'D'], ['A', 'E']]
testE2 = [['A', 'B'], ['A', 'C'], ['B', 'D'], ['A', 'C']]
testE3 = [['A', 'B'], ['A', 'C'], ['B', 'D'], ['D', 'C']]
testE4 = [['A', 'B'], ['A', 'C'], ['B', 'G'], ['C', 'H'], ['E', 'F'], ['B', 'D']]
test0 = [['B', 'D'], ['D', 'E'], ['A', 'B'], ['C', 'F'], ['E', 'G'], ['A', 'C']]
test1 = [['A', 'B'], ['A', 'C'], ['B', 'G'], ['C', 'H'], ['E', 'F'], ['B', 'D'], ['C', 'E']]

soln = SolutionTree()
print(soln.sexp(test0) == '(A(B(D(E(G))))(C(F)))')
soln2 = SolutionTree()
print(soln2.sexp(test1) == '(A(B(D)(G))(C(E(F))(H)))')
### old
## https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=690296&extra=page%3D1%26filter%3Dsortid%26sortid%3D311%26searchoption%5B3088%5D%5Bvalue...
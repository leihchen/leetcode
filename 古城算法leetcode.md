Java: [LRU cache](https://medium.com/@krishankantsinghal/my-first-blog-on-medium-583159139237)

Python: [OrderDict](https://docs.python.org/3/library/collections.html#ordereddict-objects)

[经典考题](https://www.youtube.com/watch?v=qNRKPnOaUQE&list=PLbaIOC0vpjNUg3b9zH14ikdknckN9hMMl) 1. calculator 2. number of islands

10e4-5 => nlogn

10e3 => n^2

10e8 => n

## DP 1 game theory: 

Stone Game I~IV

模版：

2player game 双方对称, 用dp计算先手玩家胜手数

$dp[i] = sum(score[i:j]) - dp[j]$

取i:j, 把交给对手, punish 对手的dp[j] (先后手反转，胜手数反转)

可使用prefix sum 或 suffix sum来优化sum

```python
def Solution():
  def StoneGame(self, n, nums):
    # recursive way
    def helper(n):
      @lru_cache(None)  # auto memo in recursive call
      if n < 0: return 0  # base case
      optimal = float('inf')  # find optimal in DP formula
      for i in ...:
        optimal = min(help(n-i), optimal)  # recursive call
      dp[i] = nums[i] + optimal
			return dp[i]
   return helper(n)
		# iterative way 略
```



## DP 2

```python
# kadane algo
# 53. Maximum Subarray
init:
  max_so_far = -inf
 	max_ending_here = -inf
for each i in a:
  max_ending_here += a[i]
  max_so_far = max(max_so_far, max_ending_here)
  max_ending_here = max(max_ending_here, 0)
return max_so_far
```

## DP 3 双序列

1143 Longest Common Subsequence

```python
f[i][j] =
					when text1[i] == text2[j]		f[i-1][j-1]
  				otherwiese									max(f[i-1][j], f[i][j-1])
```

97 Interleaving String

```python
f[i][j] =  	f[i-1][j] if 	s1[i-1] == s3[i+j-1]
						f[i][j-1] if 	s2[j-1] == s3[i+j-1]
  					otherwise			False
```

72 Edit Distance (Converting word1 to word2, this operation is sysmetric) 

```python
f[i][j] = if word1[i-1] == word2[j-1]	dp[i][j] = dp[i-1][j-1]
					else 												1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])  # 删， 增， 改
```

115 Distinct Subseq

```python
f[i][j] = # direction don't care
					f[i-1][j-1] + f[i-1][j] when A[i] == B[j] (take)
  				f[i-1][j] otherwise (skip) 
```

10 Regular Expression ( . * no parenthesis)

```python
f[string][pattern]

f[empty][empty] = True; f[i][empty] = False; pairs of zero-matching * against empty string
for j in (2, len(pattern) + 1):  # first char of pattern shouldn't be *
  f[0][j] = f[0][j-2] and pattern[j-2] == '*'
  # 				^ last pair of * gets us to empty string
  #												^ this pair is a empty star pair

f[i][j] =   # we need to check B[j-1] if B[j] is *, transition is different when other direction
					when p[j] == '.' or s[i] == p[j]			f[i-1][j-1] 
  				when p[j] == '*' 
    												when p[j-1] = s[i] 	f[i-1][j] (consume one char)
      											otherwise						f[i][j-2] (zero match star)
```

44 Wildcard Matching (? * wild-wild *)

```python
f[string][pattern]

f[empty][empty] = True; f[i][empty] = False; pairs of zero-matching * against empty string
for j in (1, len(pattern) + 1):  # first char of pattern shouldn't be *
  f[0][j] = f[0][j-1] and pattern[j-1] == '*'
  # 				^ last pair of * gets us to empty string
  #												^ this pair is a empty star pair
f[i][j] =   
					when p[j] == '?' or s[i] == p[j]			f[i-1][j-1] 
  				when p[j] == '*' 											f[i-1][j] or f[i][j-1]
    												
```

1312 Min insertion to make a string palindrome 

= len(s) - LCS(s, s.reverse())

516 Longest Palindromic Subsequence

= LCS(s, s.reverse())

1216 Valid Palindrome III

= 

## DP 4 Knapsack: 

模版：

```python
def knapsack():
    int [N+1][W+1]
    dp[0][..] = 0
    dp[..][0] = 0
    for i in [1..N]:
    	for w in [1..W]:
    		dp[i][w] = max(物品i放入, 物品i不放入)
    return dp[N][W]
```

```c++
// note that item list is unordered
// treat them independently: dp[j] means max value with j space left
// for each item, update dp[j] whose remaining space is valid 
int knapsack(int val[], int wt[], int n, int W)
{
    int dp[W+1];
    memset(dp, 0, sizeof(dp));
    for(int i=0; i < n; i++) 
        for(int j=W; j>=wt[i]; j--)
            dp[j] = max(dp[j] , val[i] + dp[j-wt[i]]);  // dependence requires a reserved order 
    return dp[W];
}
```

**0-1 knapsack: solves # of sub-arrays summing up to a value** 

```java
// find number of subset of an unordered list s.t. sum(subset) = target
private int subsetSum(int[] nums, int S){
    int[] dp = new int[S+1];
    dp[0] = 1;
    for (int i = 0; i < nums.length; i++)
        for (int j = S; j >= 0; j--)
            if (j - nums[i] >= 0) dp[j] += dp[j - nums[i]];
    return dp[S];
}
```

```python
# find number of subset of an unordered list s.t. sum(subset) = target
def subsetSum(nums, S):
    dp = [0] * (S+1)  # dp[i] is the max number of ways to reach i
    dp[0] = 1
    for n in nums:
        for j in range(S + 1)[::-1]:
            if j - n >= 0:
                dp[j] += dp[j - n]  # the new way to get sum of j is to include n; and there're d[j - n] number of ways to do this
    return dp[-1]
```

```python
# Coin Change MinCount
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount+1)
        dp[0] = 0
        for coin in coins:      
            if coin < amount: dp[coin] = 1
            for i in range(1, amount+1):
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i-coin] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1
```

```python
# 518. Coin Change 2 
# Coin Change Combinations
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:					# <-- the order
            for i in range(1, amount + 1):  # <--
                if i - coin >= 0:
                    dp[i] += dp[i-coin]
        return dp[-1]
```

```python
# Coin Change Permutation
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target+1):		# <-- the order
            for num in nums:				# <--
                if i - num >= 0:
                    dp[i] += dp[i-num]
        return dp[-1]
```

## DP 7 坐标类



[416. Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)

[474. Ones and Zeroes](https://leetcode.com/problems/ones-and-zeroes/)

[494. Target Sum](https://leetcode.com/problems/target-sum/)



## DFS, Back Tracking

```python
# backtracking
result = []
def bt(路径, 选择列表):
    if 满足条件:
        result.add(路径)
        return
   	for 选项 in 选择列表:
        make
        bt(路径, 选项列表)
        unmake
```

tree traversal (iterative)

permuation I & II, combination I & II, subsets I & II

```python
# human way of generating all subsets, 
# building from a empty set: 
# from left to right for each element, go to contain it 
# recursive call starting from the next position
# go not contain it 
def subsets(nums):
    n = len(nums)
    res = []
    def bt(start, tmp):
        res.append(list(tmp))   # res List[List[int]];
        for i in range(start, n):
            tmp += [nums[i]]
            bt(i + 1, tmp)
            tmp.pop()
    bt(0, [])
    return res
# if duplicates exists, sort it and skip duplicated elements
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums = sorted(nums)
        n = len(nums)
        def bt(start, tmp):
            res.append(list(tmp))
            for i in range(start, n):
                if i != start and nums[i-1] == nums[i]:
                    continue
                tmp.append(nums[i])
                bt(i + 1, tmp)
                tmp.pop()
        bt(0, [])
        return res
```

```python
# similar way of bt, require that the len of array should be n
# don't choose an element if already included
	def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        def bt(tmp):
            if len(tmp) == n:
                res.append(list(tmp))
            for i in nums:
                if i in tmp: continue
                tmp.append(i)
                bt(tmp)
                tmp.pop()
        bt([])
        return res
# if duplicate exsits? sort ! 
# if z appear more than once, use z_1 then z_2 then z_3 
# only add to list and call if z_i-1 is used --->
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        nums = sorted(nums)
        used_ = [False for _ in range(n)]
        def bt(tmp, used):
            if all(used):
                res.append(list(tmp))
            for i in range(n):
                if used[i] or (i != 0 and nums[i-1] == nums[i] and not used[i-1]):  # <---
                    continue
                used[i] = True
                tmp.append(nums[i])
                bt(tmp, used)
                used[i] = False
                tmp.pop()
        bt([], used_)
        return res
# combination, similar to subset?. only that require k elements instead
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        def bt(start, tmp):
            if len(tmp) == k:
                res.append(list(tmp))
                return
            for i in range(start, n+1):
                tmp.append(i)
                bt(i + 1, tmp)
                tmp.pop()
        bt(1, [])
        return res
```

Sudoku, NQueen

```python
# similar to sieve of eratosthenes
# ban different location in a dfs run
def solveNQueens(self, n: int) -> List[List[str]]:
        res = []
        def dfs(queen: list, xy_diff, xy_sum):
            # queen[i] is the 2nd-dim(y) location of the queen
            # after putting a queen at (x, y), any location (i, j) s.t. i+j = x+y OR i-j = x-y OR j = y is banned
            x = len(queen)
            if x == n:
                res.append(queen)
                return
            for y in range(n):
                if y not in queen and x+y not in xy_sum and x-y not in xy_diff:
                    # don't backtrace, each dfs call is using a copy of the lists
                    # if a thread failed to sovle, it just stops by not calling further dfs
                    dfs(queen + [y], xy_diff + [x-y], xy_sum + [x+y])
        res = []
        dfs([],[],[])
        retval = []
        # output formatting, trivial 
        for sol in res:
            tmp = []
            for y in sol:
                tmp.append('.' * y + 'Q' + '.' * (n-y-1))
            retval.append(list(tmp))
        # print(retval)
        return retval
```



## DFS, Tree traversal

```python
def inorder(root):
  res, stack = [], []
  while root and stack:
    while root:
      stack.append(root)
      root = root.left
   	if not stack:
      return res
   	node = stack.pop()
    res.append(node.val)
    root.node.right
def preorder(root):
  if not root: return []
  res, stack = [], [root]
  while stack:
    node = stack.pop()
    res.append(node.val)
    if node.left:
      stack.append(node.left)
    if node.right:
      stack.append(node.right)
'''def postorder(root):
  res, stack = [], []
  while True:
    while root:
      if root.right:
        stack.append(root.right)
      stack.append(root)
      root = root.left()'''
```



```python
# \remake 
def preorder(root):
  res, stack = [], [root]
  while stack:
    root = stack.pop()
    res.append(root.val)
    if root.right: stack.append(root.right)
    if root.left:  stack.append(root.left)  # stack, next pop will be left
  return res

def inorder(root):  # inorder of BST is sorted ASC
    res, stack = [], []
    while root or stack:
        while root: 
            stack.append(root)
            root = root.left
        root = stack.pop()
        res.appned(root.val)
        root = root.right

def postorder(root):  
  # the post order is done by a reverse trick, following preorder way
  # root, right, left ==rev==> left, right, root
  res, stack = deque([]), [root]
  while stack:
    root = stack.pop()
    res.appendleft(root.val)  # always add to front
    if root.left: stack.append(root.left)
    if root.right: stack.append(root.right)
  return res
```

## 

[Lint \551. Nested List Weight Sum](https://www.lintcode.com/problem/nested-list-weight-sum/description)

```python
"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation

class NestedInteger(object):
    def isInteger(self):
        # @return {boolean} True if this NestedInteger holds a single integer,
        # rather than a nested list.

    def getInteger(self):
        # @return {int} the single integer that this NestedInteger holds,
        # if it holds a single integer
        # Return None if this NestedInteger holds a nested list

    def getList(self):
        # @return {NestedInteger[]} the nested list that this NestedInteger holds,
        # if it holds a nested list
        # Return None if this NestedInteger holds a single integer
"""


class Solution(object):
    # @param {NestedInteger[]} nestedList a list of NestedInteger Object
    # @return {int} an integer
    def depthSum(self, nestedList):
        # Write your code here
        def dfs(nlist, depth) -> int:
            if not nlist:
                return 0
            res = 0
            for i in nlist:
                if i.isInteger():
                    res += depth * i.getInteger()
                else:
                    res += dfs(i.getList(), depth+1)
            return res
        if not nestedList:
            return 0
        return dfs(nestedList, 1)
```

```python
class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = nestedList[::-1]
        
    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop().getInteger()
        
    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:
            top = self.stack[-1]
            if top.isInteger():
                return True
            self.stack = self.stack[:-1] + top.getList()[::-1]
        return False
```

[\448. Inorder Successor in BST](https://www.lintcode.com/problem/inorder-successor-in-bst/discuss)

```python
def inorderSuccessor(self, root, p):
        # write your code here
        # O(n) Ignore BST property:
        # keep a prev ptr to keep track of p, then we can search of node whose's prev is p
        # by inorder traversal
        curr, prev, stack = root, None, []
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left 
            # curr = None
            curr = stack.pop()
            if prev and prev.val == p.val:
                return curr
            prev = curr
            curr = curr.right
        return None
        # O(h) 
        def findmin(p):
            while p.left:
                p = p.left
            return p
        if not root or not p: return None
        if p.right:
            return findmin(p.right)
        else:
            ans = None
            while root:
                if p.val == root.val:
                    return ans
                if p.val > root.val:
                    root = root.right
                else:
                    ans = root
                    root = root.left
            return None
```

[\99. Recover Binary Search Tree](https://leetcode.com/problems/recover-binary-search-tree/)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        pred = None
        wrong = [None] * 2
        def inorder(root):
            nonlocal pred, wrong
            if not root: return
            # traverse tree and set predecessor
            left = inorder(root.left)
            # login goes here
            # if we don't have more mistakes in the following traversal, then pred and root are both wrong
            # else, only pred is wrong
            if pred and pred.val >= root.val:  
                if wrong[0] and wrong[1]:
                    wrong[1] = root
                else:
                    wrong = [pred, root]  
            pred = root
            right = inorder(root.right)
        
        inorder(root)
        wrong[0].val, wrong[1].val = wrong[1].val, wrong[0].val
```



## BFS

```python
def bfs(start: Node, target: Node) -> int:
    q, visited, step = deque([start]), set([start]), 0
   	while q:
        sz = len(q)
        for i in range(sz):
            cur = q.popleft()
            if cur == target:
                return step
            for x in cur.neighbor():
                if not x in visited:
                    q.append(x)
                    visited.add(x)
         step += 1
```

Bidirectional BFS in search graph:

```python
def twoEndBfs(start: Node, target: Node) -> int:
    beginQ, endQ, visited, step = deque([start]), deque([end]), set([start]), 0
    while beginQ and endQ:
        nextQ = deque([])
        sz = len(beginQ)
        for i in range(sz):  # 展开
            cur = beginQ.popleft()
            for x in cur.neighbor():
                if x in endQ:  # endQ and startQ intersects
                    return step + 1
                if not x in visited:
                    nextQ.append(x)
        if len(endQ) < len(nextQ):  # performe reverse BFS from end if it's better
        	beginQ = endQ
            endQ = nextQ
        else:
            beginQ = endQ
        step += 1
    return 0
```

Topological Sort:

```python
def topologicalSort(graph dag):
    t = []  # dag in topo order
    q, indegree = deque([]), dict()  # stack, queue, whatever order
    for u, v in dag.edge():
    	indegree[v] += 1
    q.add([i for i in indegree if indegree[i] == 0])
    while q:
        cur = q.popleft()
        for nei in cur.neighbor():
            indegree[nei] -= 1
            if indegree[nei] == 0:
            	t.append(nei)
                q.append(nei)
	return t  # if len(t) != dag.vertex() then some cycle of isolated component exists
```

$\$ graph representation adj list

```python
# adjcency list easy way
graph = defaultdict(list)  # Map<Node, List<Node>>
for u, v in edge:
    graph[u].append(v)
# adjcency list formal way
class AdjNode:
    def __int__(self, data):
        self.vertex = data
        self.next = None
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [None] * self.V
    def add_edge(self, src, dest):
        node = AdjNode(dest)
        node.next = self.graph[src]
        self.graph[src] = node
        ## if undirected         
        # node = AdjNode(src) 
        # node.next = self.graph[dest] 
        # self.graph[dest] = node 
```



3.  





## Binary Search 基础算法II

$mid = left + (right - left) / 2$: avoid overflow

```python
s = 0
e = len - 1
while(s <= e):
    mid = s + (e - s) // 2
    if(nums[mid] < target):
        start = mid + 1
    else:
        end = mid - 1
# after termination end and start ptr are swapped, end = start - 1
```

[278. First Bad Version](https://leetcode.com/problems/first-bad-version/)

[4. Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) **HARD**

[410. Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/) **HARD** 二分法猜答案

[300. Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence)  bisect 构造

## Disjoint set/union find

Tree like structure that only cares about parent (group).

Used in dynamic connectivity problem.

```python
# easy
class DisjointSet():
    def __init__(self, n):
        self.parent = [i for i in range(n)]
    def find(self, x):
        if self.parent[x] != x:
            # simple path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
   	def union(self, x, y):   
        if self.find(x) == self.find(y): return
        parent[self.find(x)] = self.find(y)

# union by rank; worst case tree height is O(logN)
# runtime is O(log*N)
class DisjointSet:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [1] * n
        
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
   	
    def union(self, x, y) -> bool:
        rootx, rooty = self.find(x), self.finb(y)
        if rootx == rooty: return False
        if self.rank[rootx] < self.rank[rooty]:
            self.parent[rootx] = self.parent[rooty]
       	elif self.rank[rootx] > self.rank[rooty]:
            self.parent[rooty] = self.parent[rootx]
        else:
            self.parent[rootx] = rooty
            self.rank[rooty] += 1
        return True
    
class DisjointSet():
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1] * n  
    def find(self, x):
        if self.parent[x] != x:
            # simple path compression
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
   	def union(self, x, y):   
        # weight optimized: to reduce the number of parent changing in path compression, always merge small subtree into large one
        rootx , rooty = self.find(x), self.find(y)
        if rootx == rooty: return False  # x and y already in one set, loop detected
        if self.size[rooty] <= self.size[rootx]:
            self.parent[rooty] = rootx
            self.size[rootx] += self.size[rooty]
        else:
            self.parent[rootx] = rooty
            self.size[rooty] += self.size[rootx]
        return True  #x and y unioned successfully
```

Each Find and Union has O(log* n) runtime, where n is the size of union find. Log* is iterative logarithm, which is close to amortized O(1). 

[Number of Provinces](https://leetcode.com/problems/number-of-provinces)  

```python
# Amortized O(n^2). Each element in isConnected call union/find.
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        dsu = DSU(n)
        for i in range(n):
            for j in range(i):
                if i == j or isConnected[i][j] == 0: continue
                dsu.union(i, j)
        res = 0 
        for i in range(n): 
            if dsu.find(i) == i: 
                res += 1
        return res
                
class DSU: 
    def __init__(self, n):
        self.p = [i for i in range(n)]
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self,x , y):
        if self.find(x) == self.find(y): return
        self.p[self.find(x)] = self.find(y)
    
```

[261. Graph Valid Tree](https://leetcode-cn.com/problems/graph-valid-tree/)

```python
class DisjointSet():
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootx, rooty = self.find(x), self.find(y)
        if rootx == rooty: return False
        if self.size[rooty] <= self.size[rootx]:
            self.parent[rooty] = rootx
            self.size[rootx] += self.size[rooty]
        else:
            self.parent[rootx] = rooty
            self.size[rooty] += self.size[rootx]
        return True
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        ds = DisjointSet(n)
        for u, v in edges:
            if not ds.union(u, v):
                return False
        root = ds.find(0)
        return all([ds.find(i) == root for i in range(1, n)])
```



## interval overlap 扫描线、数飞机

use a number (or PQ) to keep track of number of planes on sky

sort airplanes based on start time (in some problem end time/ start ASC end DESC) 

zigzag traverse the up/down list

```
# https://www.lintcode.com/problem/number-of-airplanes-in-the-sky/
def countOfAirplanes(self, airplanes):
    # write your code here
    up, down = [], []
    for i in airplanes:
        up.append(i.start)
        down.append(i.end)
    up.sort()
    down.sort()
    i, j = 0, 0
    plane = res = 0
    while i < len(up) and j < len(down):
        if up[i] >= down[j]:
            plane -= 1 
            j += 1
        else:
            plane += 1
            res = max(res, plane)
            i += 1
    return res
# https://leetcode.com/problems/insert-interval/
# https://leetcode.com/problems/merge-intervals/
```

![lc1272-3](/Users/harddrive/Documents/GitHub/leetcode/古城算法leetcode.assets/lc1272-3-4097246.png)

![lc1272-2](/Users/harddrive/Documents/GitHub/leetcode/古城算法leetcode.assets/lc1272-2-4097246.png)

![lc1272-1](/Users/harddrive/Documents/GitHub/leetcode/古城算法leetcode.assets/lc1272-1-4097246.png)

```python
# VIP 1272 remove interval
def removeInterval(intervals, toBeRemoved):
  res = []
  for i in intervals:
    if i[1] <= toBeRemoved[0] or i[0] >= toBeRemoved[1]:  # no overlap
      res.append(i)
    else:
      if i[0] < toBeRemoved[0]:  # left end extra remaining
        res.append([i[0], toBeRemoved[0]])
      if i[1] > toBeRemoved[1]:  # right end extra remaining
        res.append([toBeRemoved[1], i[1]])
  return res
#https://leetcode.com/problems/the-skyline-problem/
```

TODO: 1229, 986, 759

## Monotonic Stack

[402 remove K digit](https://leetcode.com/problems/remove-k-digits/): **"从左到右 选大的删除"**

```python
# next greater element Ex: [2,1,2,4,3] -> [4,2,4,-1,-1]
# add element in reverse order into a stack. Earlier elements can find the CLOSEST 'nexts'
# when we find an elem that is small and far, it can't be part of the sol
# Ex [x, x, 4, 3, 5, y]
#    <---------------- when adding 4, 3 can be removed; after this step stack = [4, 5] 
# notice that elements smaller can be removed from the stack in order to keep a ASC stack
# ASC stack: when popping next elem from stack it is always larger
# in this case, if the popped element is not large enough, we can keep popping to look for a larger element that is far away

def nextGreaterElement(nums): 
  n = len(nums)
  stack, res = [], [0] * n
  for i in range(n)[::-1]:
    while stack and nums[i] >= stack[-1]: stack.pop()
    res[i] = stack[-1] if stack else -1
    stack.append(nums[i])
  return res
```

496 Next Greater Element I 

Lint 1060. Daily TemperaturesFollow: store index instead of actual value

Lint 834. Remove Duplicate Letters (unique lexicographical order of a string). Look forward instead afterward. A small char can replace a large char as long as the large one appear later

[84 Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/) ![](/Users/harddrive/Downloads/lc85.png)

## Mononotic double ended queue

```python
# 239 Sliding window maximum 
# https://www.lintcode.com/problem/362/
# Optimization 1: Heap, lazy removal sliding window  O(nlogn)
# keep a maxheap to keep track of the maximum value of scanned windows
# instead of removing left ptr element, we discard head of max heap if it's too far away
def maxSlidingWindow(nums, k):
    heap = []
    res = []
    for i in range(k):
        heapq.heappush(heap, (nums[i], i))
    res.append(heap[-1][0])
    for i in range(k, len(nums)):
        heapq.heappush(heap, (nums[i], i))
        while heapq[-1][1] <= i - k:
        	heapq.pop(-1)
        res.append(heap[-1][0])
        
# consider case: [1, 2, 3, 4]; k = 3, 
# [1, 2, 3] => 3
# [2, 3, 4] => 4, notice that '2' don't need to be checked
# consider case: [1, 1, 1, 1, 1, 4, 5], k = 6
# [1, 1, 1, 1, 1, 4] => 4
# [1, 1, 1, 1, 4, 5] => 5, notice that smaller elements can be thrown away ==> DESC Montonic Deque
	def maxSlidingWindow(nums, k):
            n = len(nums)
            q = collections.deque([])  # index
            res = []
            for i in range(n):
                while q and i - q[0] >= k: q.popleft()  # lazy remove
                while q and nums[q[-1]] <= nums[i]: q.pop()
                q.append(i)
                if i - k + 1 >= 0: res.append(nums[q[0]])
            return res
```

```python
# 队列的最大值
# https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/
# design a queue with offer, pop, and get_max in O(1) 
# get_max returns the max value in queue
class MaxQueue:

    def __init__(self):
        self.queue = collections.deque([])
        self.max_queue = collections.deque([])

    def max_value(self) -> int:
        if not self.max_queue:
            return -1- 
        return self.max_queue[0]

    def push_back(self, value: int) -> None:
        self.queue.append(value)
        while self.max_queue and self.max_queue[-1] < value:
            self.max_queue.pop()
        self.max_queue.append(value)

    def pop_front(self) -> int:
        if not self.queue:
            return -1
        if self.max_queue[0] == self.queue[0]:
            self.max_queue.popleft()
        return self.queue.popleft()

```





## Sort

```python
# quicksort, best case T(n) = O(n) + 2 O(n/2) + ...= nlgn
# worst case T(n) = T(n) + T(n-1) + ... = n^2
def quickSort(arr, low, high):
    if low >= high: return
    pv = partition(arr, low, high)
    quickSort(arr, low, pv-1)
    quickSort(arr, pv+1, high)

def partition(arr, low, high):
    wall = low
    pivot = arr[high]
    for i in range(low, high):
        # make sure left of wall is nonstrictly smaller than pivot
        if arr[i] <= pivot:
            arr[wall], arr[i] = arr[i], arr[wall]
            wall += 1
    arr[high], arr[wall] = arr[wall], arr[high]

```

## Divide and conquer

```python
# inversion pairs: 
# find all the pairs st. nums[i] > num[j] and i < j
# a large number that has a small index
# 3 1 2 -> 31, 32 
# 8 4 2 1 -> 84, 82, 81, 42, 41, 21
# start with merge sort, we count the number of such pairs in the merging step
def mergesortCount(arr, l, r):
	cnt = 0
  if l < r:
    m = (l + r) // 2
    count += mergesortCount(arr, l, m)
    count += mergesortCount(arr, m+1, r)
    count += mergeCount(arr, l, m, r)
def mergeCount(arr, l, m, r):
  left = arr[l:m+1]
  right = arr[m+1:r+1]
  i = j = k = 0
 	swap = 0
  while i < len(left) and j < len(right):
    if left[i] <= right[j]:
      arr[k] = left[i]
      k += 1; i += 1
    else:
      arr[k] = right[j]
      k += 1; j += 1
      swap += m - (l + r) + 1
  return swap
# master theorem
T(n) = a T(n/b) + O(n^c)
c^crit = log_b(a)
1. c < c_crit, leaf-heavy => O(n^c_crit)
2. c = c_crit, comparable -> rewrite f(n) part into O(n^c log^k n) => O(n^crit log^(k+1) n)
3. c > c_crit, root_heavy => O(n^c)

1.T(n) = 8T(n/2) + 1000n^2 => O(n^3)
2.T(n) = 2(T/2) + 10n => O(nlogn)
3.T(n) = 2(T/2) + n^2 => O(n^2)
```



## Sliding Windows

**Sliding Windows? Yes & No: 209 <=> 862**

https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/ 

Sliding windows/ two ptr is incorrect:

[84,-37,32,40,95] k=167 : **Right ptr goes to end to find the first valid interval.** right ptr can never reach a valid interval if too much negative exist, Expected: [32, 40, 95], Actual: None

[-28,81,-20,28,-29] k=89 : can we discard negative values? No, [81, -20, 28] is a valid interval

==> "we want to include negative numbers as long as we can know for sure that in future we have enough positive number(s) that can overcome the deficit" 

sum of subarray => focus on prefix sum of the original array

 why Monotonic Deque: we want to find P[y] - P[z] >= k while minimize y - z

Assert we only need to keep track of increasing indices: i1, i2, ..., in s.t. P[i1] < P[i2] < ... < P[in]

Consider a contradiction (non-increasing index x s.t. x < in but P[x] >= P[in]). We need to show that x can never be a optimal soln':

If P[y] - P[x] >= k, then P[y] - P[in] also >= k. But min(y-x, y-in) = y-in, therefore maintaining non-increasing x is useless.

Situation 

**Sliding Windows? Yes & No: 2062 <=> [subarray with elements occuring at least twice](https://www.geeksforgeeks.org/count-subarrays-having-each-distinct-element-occurring-at-least-twice/)**

![2cd3551e-c4de-46c8-98af-f323fb7574f8_1636260773.946565](/Users/harddrive/Documents/GitHub/leetcode/古城算法leetcode.assets/2cd3551e-c4de-46c8-98af-f323fb7574f8_1636260773.946565-8057584.png)

```python
# given a string containing only vowels
# for each ending points of a valid subarray, we can always define a right bound st s[i:j] is valid subarray
# where i in [0, bound]
class Solution:
    def countVowelSubstrings(self, word: str) -> int:
        v = set('aeiou')
        def valid(d):
            return all([d[vv] >= 1 for vv in v])
        def allfive(word):
            res = left = 0
            n = len(word)
            d = defaultdict(int)
            for i in range(n):
                d[word[i]] += 1
                while d[word[left]] > 1:   
                    d[word[left]] -= 1
                    left += 1
                if valid(d): 
                    res += left + 1
            return res  
        
        ans = 0
        prev = -1
        for i in range(len(word)):
            if word[i] not in v:
                if prev == -1:
                    prev = i
                    continue
                else:
                    ans += allfive(word[prev+1: i])
                    prev = i
        ans += allfive(word[prev+1: len(word)])
        return ans
```

```python
# Sliding window not applicable 
# weak optimization from O(N^3) -> O(N^2)

# given a string containing only elements with cnt > 2
# for ending points of a valid subarray, a bound could not be defined
# Ex: 2 3 3 3 2;    2 3 3 3 2
#       ^   ^ 		^       ^
#       |   |       |       |
# left pointer don't behave like sliding window
from collections import defaultdict
def func(A):
    result = 0
    for i in range(0,len(A)):
        counter = 0
        hashmap = defaultdict(int)
        for j in range (i, len(A)):
            hashmap[A[j]] += 1
            if hashmap[A[j]] == 2:
                counter += 1
            if counter != 0 and counter == len(hashmap):
                result += 1
    return result
```

Nature of sliding windows is two ptr, where one part don';'t need to reset to 0 index

can we safely remove the left ptr and continue? The left element is not part of future correct answer

[386. Longest Substring with At Most K Distinct Characters](https://www.lintcode.com/problem/longest-substring-with-at-most-k-distinct-characters/description)

```python
def lengthOfLongestSubstringKDistinct(s: str, k: int) -> int:
  def invalid(hashmap, k):
    # valid or invalid, determine if hashmap defines a valid window
    # valid: trim left to find a smallest window that is still valid
    # invalid: trim left to find a largest sub-window that is valid 
    # Valid Ex: len(hashmap.keys()) == k; each all(haspmap.values() >= k); etc 
    return len(hashmap) > k
  hashmap = dict()
  left, res = 0, 0
  for i in range(len(s)):  
    c = s[i]
    if c in hashmap: hashmap[c] += 1
    else: hashmap[c] = 1
    while invalid(hashmap, k):     # <- record the result here 'while valid trim to minimize'
      left_c = s[left]
      hashmap[left_c] -= 1
      if hashmap[left_c] == 0:
        del hashmap[left_c]
      left += 1
    res = max(res, i - left + 1)  # <- record the result here 'while invalid trim'
  return res
# follow up: https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/discuss/80044/Java-O(nlogk)-using-TreeMap-to-keep-last-occurrence-Interview-%22follow-up%22-question!
```

1.  exact K -> atMost(K) - atMost(K-1), handles duplicates well

2.  字母类可以尝试枚举 # of distinct = 1, 2, 3, ...., 26

3.  hard 考虑 单调队列 (unseen)

## Prefix Sum

### 2sum

```python
## https://leetcode.com/problems/subarray-sum-equals-k/
def subarraySum(self, nums: List[int], k: int) -> int:
    n = len(nums)
    sum_ = 0  # prefix sum takes sum of 0-i
    hashmap = defaultdict(int)  # hashmap[0] = (prefix_j, freq), prefix_j where j < i, freq appearance
    hashmap[0] += 1  # account for subarrays nums[0:x]
    res = 0
    for num in nums:
        sum_ += num
        res += hashmap[sum_ - k]
        hashmap[sum_] += 1
	return res
```

### rangeSum

```python
## https://www.lintcode.com/problem/range-addition/discuss
# lazy propagation
def getModifiedArray(self, length, updates):
        # Write your code here
        action = [0] * length
        for (start, end, val) in updates:
            action[start] += val
            if end + 1 < length: action[end+1] -= val
        rolling = 0
        res = []
        for i in action:
            rolling += i
            res.append(rolling)
        return res
```



## Tree

### Lowest Common Ancestor

### Serialize and Deserialize

```python
# 297 
# Use preorder traversal to do the serialization step
# The resulting string has the form
# | Root | Left Subtree | Right Subtree |
# Use recursion to deserialize that string

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        # preorder traversal
        if not root: return "#" 
        return str(root.val) + "," + self.serialize(root.left) + "," + self.serialize(root.right)
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def helper(data):
            s = data.popleft()
            if s == "#": return None
            root = TreeNode(int(s))
            root.left = helper(data)
            root.right = helper(data)
            return root
        return helper(collections.deque(data.split(',')))
```

## BST

## Trie 字典树

```python
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.is_word = False
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for c in word:
            if not node.children[ord(c) - ord('a')]:
                node.children[ord(c) - ord('a')] = TrieNode()
            node = node.children[ord(c) - ord('a')]
        node.is_word = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        for c in word:
            if not node.children[ord(c) - ord('a')]:
                return False
            node = node.children[ord(c) - ord('a')]
        return node.is_word

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for c in prefix:
            if not node.children[ord(c) - ord('a')]:
                return False
            node = node.children[ord(c) - ord('a')]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```



## Tree

Inorder traversal (left first) of BST is sorted ASC; Reversed inorder traversal (right first) of BST is sorted DESC

173 in-order iterator: 

```python
class BSTIterator:
    def __init__(self, root: Optional[TreeNode]):
        self.stack = []
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        node = self.stack.pop()
        cur = node.right
        while cur:
            self.stack.append(cur)
            cur = cur.left
        return node.val

    def hasNext(self) -> bool:
        return bool(self.stack)
```



## Stack and queue

stack: dequeue interface; queue: LinkedList interface 

```java
// haspMap that support addToVal(x) -> add x to all values in Map<int, int>
// addToKey(x) -> add to all keys in Map
class HashMapAddable{
  int keyCount; 
  int valCount;
  Map<Integer, Integer> map = new HashMap<>();
  public void insert(int[] nums){
    int insertKey = nums[0] - keyCountl;
    int insertVal = nums[1] - valueCount;
    map.put(insertKey, insertVal);
  }
  public int get(int key){
    int newKey = key - keyCountl;
    if (!map.containKey(newKey)) return -1;
    int newVal = map.get(newKey);
    return newVal + valCount;
  }
  public void addToKey(int key){ keyCount += key;}
  public void addToVal(int val){ valCount += key;}
}
```







## Number of Islands

### DFS; BFS; Union Find; Path to secribe shape; Tarjan's Algo; Distributed processing

```python
# number of island
# DFS 
dirs = [[0,1], [0,-1], [1,0], [-1,0]]
def dfs(x, y):
  if x < 0 or y < 0 or x >= len(grid) or y > len(grid[0]) or grid[x][y] != '1': return
  grid[x][y] = '0'
  for dir in dirs:
    dfs(x+dir[0], y+dir[1])
# DFS iterative
def dfs(x, y):
  if x < 0 or y < 0 or x >= len(grid) or y > len(grid[0]) or grid[x][y] != '1': return
  stack = [(x, y)]
  while stack:
    i, j = stack.pop()
    if i < 0 or j < 0 or i >= len(grid) or j > len(grid[0]) or grid[i][j] != '1': continue
    grid[i][j] = '0'
    for dir in dirs:
      q.append((i+dir[0], j+dir[1]))
# BFS
def bfs(x, y):
  if x < 0 or y < 0 or x >= len(grid) or y > len(grid[0]) or grid[x][y] != '1': return
  q = deque([(x,y)])
  while q: 
    i, j = q.popleft()
    if i < 0 or j < 0 or i >= len(grid) or j > len(grid[0]) or grid[i][j] != '1': continue
    grid[i][j] = '0'
    for dir in dirs:
      q.append((i+dir[0], j+dir[1]))
def numIslands(grid: List[List[str]]) -> int:
  cnt = 0
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      if grid[i][j] == '1':
        dfs(i, j)  / bfs(i, j)
      	cnt += 1
  return cnt
```



## 2Sum/3Sum/kSum

**3Sum**

```python
# sort to sovle duplicates
# + two pointer
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # faster than hashset, easier to deduplicate
        res = []
        nums.sort()
        n = len(nums)
        for i in range(n - 2):
            if i > 0 and nums[i-1] == nums[i]: continue  # deduplicate
            s, e = i+1, n-1
            target = -nums[i]
            while s<e:
                if nums[s] + nums[e] == target:
                    res.append([nums[i], nums[s], nums[e]])
                    s += 1
                    while s<e and nums[s] == nums[s-1]:  # deduplicate
                        s += 1
                elif nums[s] + nums[e] < target:
                    s += 1
                else:
                    e -= 1
        return res
```

**3sum-closest**: https://leetcode.com/problems/3sum-closest/submissions/ Two pointer can find closest. Hashset can't.

**[923. 3Sum With Multiplicity](https://leetcode.com/problems/3sum-with-multiplicity/)** 3SUM + combinatorics: 

![Screen Shot 2021-10-18 at 11.47.56 PM](/Users/harddrive/Documents/GitHub/leetcode/古城算法leetcode.assets/Screen Shot 2021-10-18 at 11.47.56 PM.png)

**4SUM**, depulicate is tricky

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        # 4sum -> enumerate (i, j) * 2sum
        n = len(nums)
        nums.sort()
        res = []
        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i-1]: continue  # deduplicate: prev round is the same i
            for j in range(i + 1, n - 2):
                if j > i + 1 and nums[j] == nums[j-1]: continue  # deduplicate: prev round is the same j
                sum2 = target - nums[i] - nums[j]
                s, e = j + 1, n - 1
                while s < e:
                    if nums[s] + nums[e] == sum2:
                        res.append([nums[i], nums[j], nums[s], nums[e]])
                        s += 1  # move both ptr
                        e -= 1
                        while s < e and nums[s] == nums[s-1]: s += 1  # move till not equal
                        while s < e and nums[e] == nums[e+1]: e -= 1
                    elif nums[s] + nums[e] > sum2:
                        e -= 1
                    else:
                        s += 1
        return  res
```

**4 SUM II. Loop unrolling**

```python
    # Brute force
    for (int a = 0; a < A.size(); a++) 
        for (int b = 0; b < B.size(); b++)
            for (int c = 0; c < C.size(); c++)
                for (int d = 0; d < D.size(); d++)    
                     if (A[a] + B[b] = - C[c] - D[d])
                        ...
```

it's comparing every sum of `A[a]` and `B[b]` with `C[c]` and `D[d]`. Since it's comparing every `A[a] + B[b]` with every `C[c] + D[d]`, why don't we just precompute every `A[a] + B[b]` and store those results? This'll save us a lot of time.

```python
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        n = len(nums1)
        ab = defaultdict(int)
        for i in nums1:
            for j in nums2:
                ab[i + j] += 1
        res = 0
        for m in nums3:
            for n in nums4:
                res += ab[- m - n] 
        return res
```

**[k-sum](https://www.lintcode.com/problem/89/): find number of ways only**

ways to select k elements from array to sum up to target

```python
# only find the number of ways
class Solution:
    """
    @param A: An integer array
    @param k: A positive integer (k <= length(A))
    @param target: An integer
    @return: An integer
    """
    def kSum(self, A, k, target):
        # write your code here
        n = len(A)
        # dp[m][n][q]: number of ways to get a sum of q when chosing n element from A[:m]
        dp = [[[0 for _ in range(target+1)] for _ in range(k+1)] for _ in range(n+1)]
        for i in range(n+1): dp[i][0][0] = 1
        for i in range(1, n+1):
            for j in range(1, k+1):
                for t in range(target+1):
                    dp[i][j][t] = 0
                    if t >= A[i-1]:
                        dp[i][j][t] = dp[i-1][j-1][t-A[i-1]]
                    dp[i][j][t] += dp[i-1][j][t]
        return dp[n][k][target] 
```

**[k-sum II](https://www.lintcode.com/problem/90/): find all combinations** 

```python
# Method 1: brute force subset
class Solution:  
    """
    @param A: an integer array
    @param k: a postive integer <= length(A)
    @param target: an integer
    @return: A list of lists of integer
    """
    def kSumII(self, A, k, target):
        # write your code here
        res = []
        n = len(A)
        A.sort()
        def dfs(i, tmp, sum_):
            if len(tmp) == k:
                if sum_ == target:
                    res.append(tmp)
                return
            if i == n or sum_ > target:
                return
            for j in range(i, n):
                if A[j] > target: break
                dfs(j+1, tmp + [A[j]], sum_ + A[j])
        dfs(0, [], 0)
        return res

# Method 2: recursive k-sum (layer by layer, reduce to 2 sum)  O(Nk)
class Solution:
    """
    @param A: an integer array
    @param k: a postive integer <= length(A)
    @param target: an integer
    @return: A list of lists of integer
    """
    def kSumII(self, A, k, target):
        # write your code here
        A.sort()
        return self.ksum(A, k, target)

    def ksum(self, A, k, target):
        res = []
        if k == 1:
            return [] if target not in A else [[target]]
        if k == 2:
            return self.twosum(A, target)
        # pruning trick: if smallest value is too larget or largest value too smaller, just break
        # if A[0] * k > target or A[-1] * k < target:
        #     return []
        for i in range(0, len(A) - k + 1):
            if i > 0 and A[i] == A[i-1]: continue
            subset = self.kSumII(A[i+1:], k - 1, target - A[i])
            for sub in subset:
                res.append([A[i]] + sub)
        return res

    def twosum(self, nums, target):  
        # assume sorted nums, allows duplicate
        # returns twosum combinations
        res = []
        s , e = 0, len(nums) - 1
        while s < e:
            if nums[s] + nums[e] == target:
                res.append([nums[s], nums[e]])
                s += 1
                e -= 1
                while nums[s] == nums[s-1]: s += 1
                while nums[e] == nums[e+1]: e -= 1
            elif nums[s] + nums[e] < target:
                s += 1
            else:
                e -= 1
        return res
```

## Jump Game

双向跳，BFS或双向BFS

单向跳，dp (brute force) --> minheap (NlogN) --> monotonic stack/heap (N)

## Bit Manipulation

XOR: 同0异1

A ^ B = B ^ A; A ^ A = 0; A ^ 0 = A
判断相等：a ^ b == 0

数位翻转：0 ^ 1 = 1; 1 ^ 1 = 0; eg: 10100001 ^ (1 << 5) = 10000001 flip 5th bit

二进制中 1 的个数是奇数还是偶数：10100001: XOR到一起=1， 1=>奇数个1; 0=>偶数个1

num & (num - 1): 将二进制num中最右的 ‘1’ 变为 ‘0’ (eg: 011 => 010; 100 => 000)

## 离线query

sort query + active region

## Catches

if two string has the same length -> sum of Counter is the same. If the Counter is different, then some element must be negative. 242

a string with n pair of balanced parentheses: as we add new char into the str, #(lp) is always < n AND #(rp) is always < #(lp)

KMP is a plus

[Dutch National Flag Algo](https://www.geeksforgeeks.org/sort-an-array-of-0s-1s-and-2s/) 75

```python
# eratosthenes prime finding
def eratosthenes(n):
    IsPrime = [True] * (n + 1)
    for i in range(2, int(n ** 0.5) + 1):
        if IsPrime[i]:
            for j in range(i * i, n + 1, i):
                IsPrime[j] = False
    return [x for x in range(2, n + 1) if IsPrime[x]]
if __name__ == "__main__":
    print(eratosthenes(120))
```

```java
// find the lead of array
public class Solution {
    public int majorityElement(int[] num) {

        int major=num[0], count = 1;
        for(int i=1; i<num.length;i++){
            if(count==0){
                count++;
                major=num[i];
            }else if(major==num[i]){
                count++;
            }else count--;
            
        }
        return major;
    }
}
```

Inorder traveral of a BST is an ASC sorted array

```python
# inorder iterative
```

Magic: 334. [Increasing Triplet Subsequence](https://leetcode.com/problems/increasing-triplet-subsequence/) https://leetcode-cn.com/problems/increasing-triplet-subsequence/solution/pou-xi-ben-zhi-yi-wen-bang-ni-kan-qing-t-3ye2/

Magic: [31. Next Permutation](https://leetcode.com/problems/next-permutation) ![next-permutation-algorithm](/Users/harddrive/Documents/GitHub/leetcode/古城算法leetcode.assets/next-permutation-algorithm.svg)

## Greedy Examples

gas station: starting point should be the next of the one that's hardest to reach

interval scheduling problem: given an array of meeting start and end time, find the max number of meeting to hold with one meeting room.  Sort end time, pick earlies end time meetings.

https://leetcode.com/problems/non-overlapping-intervals/

Suppose current earliest end time of the rest intervals is `x`. Then available time slot left for other intervals is `[x:]`. If we choose another interval with end time `y`, then available time slot would be `[y:]`. Since `x ≤ y`, there is no way `[y:]` can hold more intervals then `[x:]`. 

```python
# 1024 AND 1326
class Solution:
    def videoStitching(self, clips: List[List[int]], time: int) -> int:
        res = st = curEnd = 0
        clips.sort()
        n = len(clips)
        i = 0
        while i < n:
            if st >= time:
                return res
            if clips[i][0] > st:
                return -1
            while i < n and clips[i][0] <= st:
                curEnd = max(curEnd, clips[i][1])
                i += 1
            res += 1
            st = curEnd
        return res if st >= time else -1
```



## 信息传递

backtracking is similar to DFS, one kind of brute force. (ex Nqueen problem). Usually its helper function returns void (no early quit in order to brute force all); and stores information in nonlocal variables.

257 

**Pure recursion** (bottom up)

**Backtracking** (top down) 信息简单地自上而下传递

## Graph

### Undirected Cycle Detection

[261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree)

```python
# Method 1: Union Find
# Graph format: edge list
# Ex: 
def validTree(n, edges):
    dsu = DSU(n)
    for edge in edges:
        if not dsu.union(edge[0], edge[1]): 
            return False
    return len(edges) == n-1

# Method 2: dfs, find back edge
# Graph format: adjacency list
def isCyclic(graph: defaultdict(list)):
    n = len(graph)
    visited = [False] * n
    for v in graph:
        if not visited[v]:
            if isCyclicUtil(i, -1):
                return True
    
    def isCyclicUtil(cur, parent):
        visited[cur] = True
        for nei in graph[cur]:
            if not visited[nei]:
                if isCyclicUtil(nei, cur):
                    return True
            else:
                if nei != parent:  # ignore trivial back edge between parent and cur 0 <-> 1
                    return True
    
```

### Directed Cycle Dectection (DAG)

```python
# topological sort with DFS
def isDAG(n, graph):
    visited = [0] * n
    for v in range(vertice):
        if visited[v] == 0:
            if not dfs(v):
                return False
def dfs(v):
    visited[v] = 1 # 1=visiting
    for nei in graph[v]:
        if visited[nei] == 1:
            return False
        if visited[nei] == 0:  #0=unvisited
            dfs(nei)
    visited[v] = 2  # 2=visited    
# topological sort with BFS (kahn's algo 入度表)
```

### MST

Naive Prim (faster for dense graph) O(V^2) --dense--> O(n^2)

```python
graph = [[0 for _ in range(N)] for _ in range(N)]  # AdjMatrix
visited = set()
distance = [float('inf')] * N
distance[0] = 0
for i in range(N):
  nextClosest = -1
  for j in range(N):
    if j not in visited:  # 在剩余vertex 里面 找到最小路径的vertex
      if nextClosest == -1 or distance[j] < distance[nextClosest]:
        nextClosest = j
    visited.add(nextClosest)
      for y in range(N):
        if y not in visited:
          distance[y] = min(distance[y], graph[nextClosest][y])
        
```



Prim (greddy, edge relaxation; PQ + DFS) O(ElogV)  --dense--> O(n^2 log n) 

```python
# https://www.youtube.com/watch?v=jsmMtJpPnhU
def lazyPrim(graph, n, start=0) -> int:
	# given a graph with n vertices, find the min spanning tree rooted at start
    # Prim's Algo (greedy)
    # graph = defaultdict()  # AdjList size=N, graph[i] -> (nei, weight)
    pq = []  # edge = (weight, src, dst)
    visited = set()
    def addEdge(node):
		visited.add(node)
        # iterate all outgoing edges of node
        for nei in graph[node]:
		if nei not in visited:
        	heappush(pq, (nei[1], node, nei[0]))
	addEdge(start)
  	mstEdges = []  # should be of size |V| - 1
  	mstCost = 0
  	while pq and len(mstEdges) != n-1:
    	z,x,y = heapq.pop(pq)
    	if y in visited:
      		continue
    	mstCost += z
    	mstEdges.append((z,x,y))
		addEdge(y)
    if len(mstEdges) != n - 1:
    	return None, None  # graph is not connected
	return mstEdges, mstCost
```

Kruscal (; UnionFind)

Sort all edges. Add edges from smallest weight. Ignore unnecessary edges (ie edges that connects node in the same Union Set)







## Parentheses

Basic: Rolling state

[1249](https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/): Two pass 正反向 Rolling state

[856](https://leetcode.com/problems/score-of-parentheses/submissions/): parentheses + calculator -> stack

## DFS Pruning

Assignment emuneration:

[1986](https://leetcode.com/problems/minimum-number-of-work-sessions-to-finish-the-tasks/submissions/), [1723](https://leetcode.com/problems/find-minimum-time-to-finish-all-jobs/)

**Caveat** **1**: The 'assignments array' to keep track of assignments is not necessary. In dfs, the array will grow and shrink but only one array is used at execution. So we can modify the array -> do dfs -> undo array. **Without** including array as variable of func dfs. 

**Caveat** **2**: common pruning:

1.  Discard invalid result when detected, Ex: `if len > res: return`
2.  In assignment for-loop, don't differentiate assignment with save value. Ex: `if j >= 1 and assignment[j] == assignment[j-1]: continue`. 
3.  Skip same value, Ex: premutation II
4.  Sort tasks in desc order. Assign large tasks first. 

## Basic Calculator 

**Only + -**: Push all into stack with sign, sum up to answer

**All + - * /**: Since * / have higher level, do calculator when pushing. Ex: stack.push(stack.pop() * num). And then reduced to **only + -** case.

**With ( )**: Recursion in **all + - * /**

Caveat: in py, -4 // 3 = -2 , int(-4 / 3) = -1. `//` is FLOOR division not integer division



## TODO

quickselect 215 largest Kth element

bucket sort 451 

dijkstra algo

## Random Useful Hard

https://leetcode.com/problems/closest-subsequence-sum/

Meet in the middle: (brute force on first and second half of a list, find result by binary searching)

## Ignored Hard

352 Data Stream as Disjoint Intervals [扫描线](https://www.youtube.com/watch?v=ihf8JjQdta0&t=1221s)

[Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack)   [Stack](https://www.youtube.com/watch?v=cV3SpacBh3M&t=11s)

## Ignored VIP

[ Meeting Scheduler](https://leetcode.com/problems/meeting-scheduler) 

[Employee Free Time](https://leetcode.com/problems/employee-free-time)

1168 

## Language Specific Structures 

**SortedDict**(): sorted dict inherits from dict to store items and maintains a sorted list of keys

Special method: 

​	peekitem(*index*=-1), popitem(*index*=-1)  -> (k, v) pair sorted by k at *index*.

**Bisect.bisect_left**(*a*, *x*, *lo*=0, *hi*=len(a)): Locate the insertion point for *x* in *a* to maintain sorted order

​	The parameters *lo* and *hi* may be used to specify a subset of the list which should be considered; by default the entire list is used. If *x* is already present in *a*, the insertion point will be before (to the left of) any existing entries. The return value is suitable for use as the first parameter to `list.insert()` assuming that *a* is already sorted.

​	The returned insertion point *i* partitions the array *a* into two halves so that `all(val < x for val in a[lo : i])` for the left side and `all(val >= x for val in a[i : hi])` for the right side.

**Itertools.groupby**()

​	`subcnt = [c, len(list(g)) for c, g in groupby(str)]`

**Zip, transpose, rotate**

​	**zip(*iterables)**: `[ [iter[0][0], iter[1][0], ...], [iter[0][1], iter[1][1]]]`

```
number_list = [1, 2, 3]
str_list = ['one', 'two', 'three']
zip(number_list, str_list)
> [(1, 'one'), (2, 'two'), (3, 'three')]
```

​	**zip(*matrix)**: transpose matrix	

​	**zip(*matrix)[::-1]**: rotate matrix (90 degree counter-clockwise). Reversing after transposing is rotating

​	To get list out of iterable: use `list(zip(*board))[::-1] `

**Count substring in string:**

​	Overlapping: `re.findall(f"(?={pattern})", word)`

​	Non-overlapping: `word.count(pattern)`


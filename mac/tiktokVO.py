# 694
# 210
# 76
# 33
# 2. 汇率转换。给你a种钱,问可以换多少的b种钱。然后有一个table是显示每个之间汇率的。
# e.x.
# data = [("USD", "JPY", 110), ("USD", "AUD", 1.45), ("JPY", "GBP", 0.0070)]
# 我们的任务是写出getRatio( data,"JPY", "AUD”,1)。 用bfs解决。
# bq (vo）
# 1. 遇到conflict怎么办？
# 2. 之前项目遇到的challenge？
# 3. 你希望做什么？
# 4. 你之前有什么mistake？‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌
# 5. 你对这个职位的期望是什么？
# most challenging project?
# 给予一个排序数组和整数k和x, 求k个最接近x的数。658
# 闯关模式的面经
# 1.  shuffle array + random pick,  median in data stream
# 2. 数组的iterator 及其变种
# 3. SD: Trending tags
# 4. Project Deep Dive + B‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌Q
# 722
# 473
# 39
# 1004
# 480. Sliding Window Median
# 953. Verifying an Alien Dictionary
# 1428. Leftmost Column with at Least a One
# 489
# total 三面 第一面就掛了
# Hackerank 45 mins
# 流程:
# 1. intro + Resume question about the project (5 mins)
# 2. given a binary search tree return sum of the path
#      4
#    /   \
#   5.   0
# /   \
# 6  7
# 456 + 457 + 40 = 953
# 3. ask question back
# 附上面試前，掃過一畝 Tiktok VO 面經 的summary
# Leetcode:
# 26 Remove Duplicates from Sorted Array
# 67 Add Binary
# 79 Word Search
# 91 Decode Ways
# 93 Restore IP Addresses
# 102 Binary Tree Level Order Traversal
# 105 Construct Binary Tree from Preorder and Inorder Traversal 124 Binary Tree Maximum Path Sum
# 163 Missing Ranges
# 207 Course schedule 有向圖找環
# 224 Basic Calculator + 變形 可以有 invalid input ex: “3 2” 227 Basic Calculator II
# 236 (就是求二叉树两点之间最短路径，本质其实就是二三流)
# 261 Graph Valid Tree
# 282 Expression Add Operators
# 295 Find Median from Data Stream
# 301 Remove Invalid Parentheses
# 328 Odd Even Linked List
# 341 Flatten Nested List Iterator
# 394 Decode String
# 490 The Maze
# 863 All Nodes Distance K in Binary Tree
# 953 Verifying an Alien Dictionary
# 1047 Remove All Adjacent Duplicates in String
# 1109 Corporate Flight Bookings
# 1235 (Research) Maximum Profit in Job Scheduling
# 1352 Product of the La‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌st K Numbers
# 1428 (follow up: if this matrix only one line and very long > Binary Search)
# Question:
# Python 是不是 pass by reference? Resume
# 基礎知識
# HTTP vs HTTP2:
# Angular VS React:
# SEO(Search Engine Optimize):
# CSRF:
# 给你一堆async promise timeout await, print出来的顺序 How to migrate project:
# safe deployment:
# 第一轮出了一个比较简单的题，二维数组从左上角向右向下走到右下角，问最小的整数和（走过的cell 相应的整数加起来）。
# 第二轮问了些工程上的东西，然后问了100坛酒怎么样用最少的士兵找出其中一‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌坛毒酒。
# 翻转链表的前k个节点，如果k大于链‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌表长度，就不翻转，返回原链表
# 力扣： 幺八三四， 二幺五
# 981
# golang concurrency实现方式
# 353
# 面试小姑娘人很好。做了两个题，都很简单。
# 第一个：binary tree maximum path sum，应该是leetcode原题，不记得题号了。
# 第二个：word search，应该也是原题，但不记得题号了。经典的‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌DFS
# 2021.11.29 一面：算法题（LeetCode 341），聊简历
# 2021.12.09 二面：算法题（CCF201812-3 CIDR合并），聊简历
# 1352
# 236
# 给一个整数N，要求输出一个N×N的矩阵，矩阵里面的值由外到内呈螺旋状递增 59
# 比如N＝3就要返回
# 1  2  3
# 8  9  4
# 7  6  5
# 再比如N＝4就要
# 1    2    3  4
# 12  13  14  5
# 11  16  15  6
# 10  9    8  7
# Kth Smallest Element in a Sorted Matrix
# 求字符串中最长不重复子串

# def intersect(left, right):
#     i = 0
#     for x in left:
#         for i in range(i, len(right)):
#             if right[i] > x:
#                 break
#             if right[i] == x:
#                 yield x
#
# left = [1,2,3,4,5,6,7,7]
# right = [2,4,5,5,7]
# for v in intersect(left, right):
#     print(v)
# def f(n):
#     if n <= 0:
#         return 0
#     return n + f(int(n/2))
# print(f(4))

from typing import *
def subStrHash(s: str, power: int, modulo: int, k: int, hashValue: int) -> str:
    def c2v(c):
        return 1 + ord(c) - ord('a')

    n = len(s)
    s = s[::-1]
    rolling = 0
    for i in range(k):
        rolling = (rolling * power + c2v(s[i])) % modulo
    res = ''
    for i in range(k, n):
        if rolling == hashValue:
            return s[i-k+1:i+1][::-1]
        rolling = (rolling - c2v(s[i - k]) * power ** (k - 1)) % modulo
        rolling = (rolling * power + c2v(s[i])) % modulo
    if rolling == hashValue:
        res = s[:k]
    return res
# print(subStrHash('fbxzaad', 31, 100, 3, 32))

from collections import defaultdict, deque
from itertools import *
class Solution:
    def __init__(self, exchanges):
        graph = defaultdict(list)
        for u, v, rate in exchanges:
            graph[u].append((v, rate))
            graph[v].append((u, 1/rate))
        self.graph = graph

    def getExchangeRatio(self, src, dst):
        q = deque([src])
        visited = {src: 1}
        while q:
            node = q.popleft()
            if node == dst:
                return visited[node]
            for nei, rate in self.graph[node]:
                if nei not in visited:
                    visited[nei] = visited[node] * rate
                    q.append(nei)
        return -1


data = [("USD", "JPY", 115.08), ("USD", "AUD", 1.42), ("JPY", "GBP", 0.0065), ("RMB", "USD", 0.16), ]
soln = Solution(data)
# print(soln.getExchangeRatio("JPY", "AUD"), 0.012)
# print(soln.getExchangeRatio("GBP", "AUD"), 1.90)
# print(soln.getExchangeRatio("RMB", "AUD"), 0.22)


class TreeNode:
    def __init__(self, x, left=None, right=None):
        self.val = x
        self.left = left
        self.right = right

def numberTreePathSum(root: TreeNode):
    res = 0
    def dfs(node, num):
        nonlocal res
        if not node.left and not node.right:
            res += num * 10 + node.val
        if node.left:
            dfs(node.left, num * 10 + node.val)
        if node.right:
            dfs(node.right, num * 10 + node.val)
    dfs(root, 0)
    return res

test = TreeNode(4)
test.left = TreeNode(5, left=TreeNode(6), right=TreeNode(7))
test.right = TreeNode(0)
# print(numberTreePathSum(test))
def isAlienSorted(words: List[str], order: str) -> bool:
    d = {c: i for i, c in enumerate(order)}
    alien_words = [[d[c] for c in word] for word in words]
    print(alien_words)
    return all(alien_words[i - 1] <= alien_words[i] for i in range(1, len(words)))
isAlienSorted(["hello","leetcode"], "hlabcdefgijkmnopqrstuvwxyz")
print([0, 6, 1, 1, 14] < [1, 6, 6, 19, 4, 14, 5, 6])


def countBinarySubstrings(s: str) -> int:
    subcnt = [len(list(g)) for _, g in groupby(s)]
    n = len(subcnt)
    res = 0
    for i in range(1, n):
        res += min(subcnt[i - 1], subcnt[i])
    return res

class Solution:
    i = 0
    def calculate(self, s: str) -> int:
        s.replace(' ', '')
        sign = '+'
        num = 0
        stack = []
        if not s: return 0
        while self.i < len(s):
            c = s[self.i]
            self.i += 1
            if c.isdigit():
                num = 10 * num + int(c)
            if c == '(':
                num = self.calculate(s)
            if not c.isdigit() or self.i >= len(s):
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
# soln = Solution()
# print(soln.calculate("1 + 1"))

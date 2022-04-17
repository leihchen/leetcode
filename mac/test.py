from typing import *

# def findPeakElement(values: List[int]) -> int:
#     # find the first decreasing point
#     n = len(values)
#     left, right = 0, n - 1
#     while left <= right:
#         mid = (left + right) // 2  # (left + right) // 2
#         if values[left] < values[mid] < values[right]:
#             left = mid + 1
#         elif values[left] > values[mid] > values[right]:
#             right = mid - 1
#         elif values[left] < values[mid]:
#             right = mid - 1
#         else:
#             left = mid + 1
#     return values[left]
# print(findPeakElement([1,2,1,3,5,6,4]))


from collections import deque, defaultdict
class TreeNode:
    def __init__(self, val='#', left=None, right=None):
        self.val = val
        self.left = left
        self.right = right








#     def tprint(self, root):
#         q = deque([root])
#         while q:
#             sz = len(q)
#             res = []
#             for _ in range(sz):
#                 cur = q.popleft()
#                 res.append(cur.val if cur else '#')
#                 if cur: q.append(cur.left)
#                 if cur: q.append(cur.right)
#             print(res)
#
# def buildLevelOrder(level):
#     def build(i):
#         if not level[i]:
#             return None
#         node = TreeNode(level[i])
#         node.left  = build(2*i+1)
#         node.right = build(2*i+2)
#         return node
#
#     return build(0)
#
# # test = [1,2,3,4,5,6,None, None, None, 7, 8, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
# # root = buildLevelOrder(test)
# # root.tprint(root)
# #
#
# def longestSubstring(s: str, k: int) -> int:
#     result = 0
#     n = len(s)
#     for i in range(0, n):
#         counter = 0
#         hashmap = defaultdict(int)
#         for j in range(i, n):
#             hashmap[s[j]] += 1
#             # print(hashmap[s[j]], s[j])
#
#             if hashmap[s[j]] >= k:
#                 print(hashmap[s[j]], s[j])
#                 counter += 1
#             if counter == len(hashmap.keys()):
#                 # print(hashmap, len(hashmap), counter)
#                 result = max(result, j - i + 1)
#     return result
#
# # print(longestSubstring('ababbc', 2))
#
# '''
# Schedule for Today:
# 3:00pm – 3:05pm Warm up, Introduction
# 3:05pm – 3:45pm Coding Question:
# 3:45pm – 3:55pm Behavioral Questions
#
#
#
#
#
# 4:00pm Q&A
#
# (1) Build a Binary Tree based on input integer array
# (2) Verify if it is a symmetric Binary Tree
#
# Input: root = [1,2,2,3,4,4,3]
# Output: true
#
#       1
#    2     2    (2   2  <= 1)
# 3   4  4   3
#
# null
#
# Input: root = [1,2,2,null,3,null,3],
#                0 1 2 3    4  5  6   7 # O(n) of time, O(1) space (stack space)
# # 1, 2, 4, 8 ...
#
# # in complete tree, node with index i in the level-order array, left: 2*i+1, right: 2*i+2
#
# Output: false
#
#        1
#    2      2
#       3      3
#
# '''
# from collections import deque
#
#
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
#
# def buildTree(nums):  # List<Integer>, int[]
#     n = len(nums)
#
#     # stack = []
#     def build(i) -> TreeNode:  # recursive # 2n
#         if i >= n:
#             return None
#         node = None
#         if nums[i]:
#             node = TreeNode(nums[i], left=build(2 * i + 1), right=build(2 * i + 2))
#         return node
#
#     root = build(0)
#     return root
#
#
# def levelOrderTraversal(root) -> bool:
#     # print in level order, if None print #
#     d = deque([root])
#     isSymc = True
#     while d:
#         size = len(d)
#         tmp = []
#         for _ in range(size):
#             node = d.popleft()
#             if node:
#                 tmp.append(node.val)
#                 d.append(node.left)
#                 d.append(node.right)
#             else:
#                 tmp.append(-1)
#         if tmp[::-1] != tmp:
#             isSymc = False
#         print(tmp)
#     return isSymc
#
#
# leftsub = TreeNode(2, left=None, right=TreeNode(3))
# rightsub = TreeNode(4, left=None, right=TreeNode(5))
# root = TreeNode(1, left=leftsub, right=rightsub)
# #        1
# #    2      4
# #       3      5
# #
# # levelOrderTraversal(root)
# test = [1, 2, 2, 3, 4, 4, 3]
# root = buildTree(test)
# levelOrderTraversal(root)
#
#
# def symmetric(root):
#     def helper(l, r) -> bool:
#         if (l and not r) or (r and not l):
#             return False
#         if l.val != r.val:
#             return False
#         return helper(l.left, r.right) and helper(l.right, r.left)
#         # verify is a is mirror of b
#         # root of l == root of r
#         # l.left and r.right is mirror
#         # l.right and r.left is mirror
#
#     return helper(root.left, root.right)
#
#
# # theta(n) = 2 * theta(n/2) + O(1)
# # O(n) - time
# # heap space O(1)
#
# '''
# def addNumbers(a,b):
#     sum = a + b
#     return sum
#
# num1 = int(input())
# num2 = int(input())
#
# print("The sum is", addNumbers(num1, num2))
# '''
#
# '''
# Earn trusts?
# Give trusts?
# '''
#
#
# # Python3 implementation of the approach
#
# # Function to print the required string
# def printString(Str1, n):
#     # count number of 1s
#     ones = Counter(Str1)['1']
#
#     # To check if the all the 1s
#     # have been used or not
#     used = False
#
#     for i in range(n):
#         if (Str1[i] == '2' and used == False):
#             used = 1
#
#             # Print all the 1s if any 2 is encountered
#             for j in range(ones):
#                 print("1", end="")
#
#         # If Str1[i] = 0 or Str1[i] = 2
#         if (Str1[i] != '1'):
#             print(Str1[i], end="")
#
#     # If 1s are not printed yet
#     if (used == False):
#         for j in range(ones):
#             print("1", end="")


# Driver code
# Str1 = "100210"
# n = len(Str1)
# printString(Str1, n)
#
# # This code is contributed
# # by Mohit Kumar
# res = []
# res.append('a' * 0)
# print(res)
# a, b = 545, 95922948
# res = 0
# for x in range(1, 31624):
#     if a <= x * (x+1) <= b:
#         res += 1
# print(res)
#
# d = TreeNode('D', TreeNode('C'), TreeNode('E'))
# b = TreeNode('B', TreeNode('A'), d)
# i = TreeNode('I', TreeNode('H'))
# g = TreeNode('G', None, i)
# f = TreeNode('F', b, g)
#
# def inOrder(root):
#     if root:
#         inOrder(root.left)
#         print(root.val)
#
#         inOrder(root.right)
# inOrder(f)

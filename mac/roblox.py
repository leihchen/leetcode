from collections import deque
from typing import *
# class Solution:
#     def minKnightMoves(self, x: int, y: int) -> int:
#         # bfs
#         q = deque([(0, 0)])
#         level = 0
#         visited = set([(0, 0)])
#         dirs = [[1, 2], [2, 1], [-1, 2], [-2, 1], [-2, -1], [-1, -2], [1, -2], [2, -1]]
#         while q:
#             sz = len(q)
#             for _ in range(sz):
#                 node = q.popleft()
#                 if node == (x, y):
#                     return level
#                 for dx, dy in dirs:
#                     new_node = (node[0] + dx, node[1] + dy)
#                     if new_node not in visited:
#                         visited.add(new_node)
#                         q.append(new_node)
#             level += 1

# lc 210 multithreading

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # wordSet = set(wordList)
        # q = deque([beginWord])
        # visited = set(q)
        # level = 1
        # while q:
        #     sz = len(q)
        #     for _ in range(sz):
        #         word = q.popleft()
        #         if word == endWord:
        #             return level
        #         for i in range(len(word)):  # [0,n-1]
        #             for replace in range(26):
        #                 new = word[:i] + chr(replace + ord('a')) + (word[i+1:] if i + 1 < len(word) else '')
        #                 if new in wordSet and new not in visited:
        #                     q.append(new)
        #                     visited.add(new)
        #     level += 1
        # return 0

        beginSet, endSet = set([beginWord]), set([endWord])
        wordSet = set(wordList)
        level = 1
        if endWord not in wordSet: return 0
        while beginSet or endSet:
            nextSet = set()
            print(beginSet, endSet)
            for word in beginSet:
                for i in range(len(word)):
                    for replace in 'abcdefghijklmnopqrstuvwxyz':
                        new = word[:i] + replace + word[i + 1:]
                        if new in endSet: return level + 1
                        if new in wordSet:
                            nextSet.add(new)
                            wordSet.discard(new)
            if len(endSet) < len(nextSet):
                beginSet = endSet
                endSet = nextSet
            else:
                beginSet = nextSet
            level += 1
        return 0
# soln = Solution()
# print(soln.ladderLength("hot", "dog", ["hot", "dog"]))

# class Solution:
def findPeakElement(nums: List[int]) -> int:
    n = len(nums)
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if mid + 1 >= n or nums[mid + 1] < nums[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return left
test = [1,1,2,1,1,1,1,1]
print(findPeakElement(test), len(test))
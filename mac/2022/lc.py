# def searchMatrix(matrix, target: int) -> bool:
    # binary search twice
    # m, n = len(matrix), len(matrix[0])
    # sm, em = 0, m - 1
    # while True:
    #     mid = sm + (em - sm) // 2
    #     if matrix[mid][0] == target:
    #         return True
    #     elif matrix[mid][0] > target:
    #         em = mid - 1
    #     else:
    #         break
    # sm = mid
    # sn, en = 1, n - 1
    # while sn <= en:
    #     mid = sn + (en - sn) // 2
    #     if matrix[sm][mid] == target:
    #         return True
    #     elif matrix[sm][mid] < target:
    #         sn = mid + 1
    #     else:
    #         en = mid - 1
    # return False
# print(searchMatrix([[1],[3]], 15))
from pprint import pprint
def findMin(nums) -> int:
    # if nums[0] < nums[-1]:
    #     return nums[0]
    start, end = 0, len(nums) - 1
    while start < end:
        mid = start + (end - start) // 2
        if nums[mid] < nums[end]:
            end = mid
        elif nums[mid] > nums[end]:
            start = mid + 1
        else:
            start += 1
        print(start, end)
    print(nums[start: end+1])
    return nums[start]
# print(findMin([10,1,10,10,10]))


def search(nums, target: int) -> int:
    # undo rotation by finding index the smaller elem
    start, end = 0, len(nums) - 1
    while start < end:
        mid = start + (end - start) // 2
        if nums[mid] > nums[end]:
            start = mid + 1
        else:
            end = mid
    # start == end == index of smallest
    # print(start, end, n)
    start, end = start, (start + len(nums))
    while start <= end:
        # print(start, end)
        end_index = end % len(nums)
        mid = (start + end) // 2
        mid_index = mid % len(nums)
        # mid = (start + (end_index - start) // 2) % len(nums)
        # print(mid)
        if nums[mid_index] == target:
            return mid
        elif nums[mid_index] > target:
            end = (len(nums) + mid - 1) % len(nums)
        else:
            start = (mid + 1) % len(nums)

    return -1
def search(nums, target: int) -> bool:
    # undo rotation
    n = len(nums)
    start, end = 0, n-1
    while start < end:
        mid = start + (end - start) // 2
        if nums[mid] < nums[end]:
            end = mid
        elif nums[mid] > nums[end]:
            start = mid + 1
        else:
            end -= 1
    print(start, end)
# print(search([1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1],2))


# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val


class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        vertex_set = dict()
        pass
# d ={}
# n4 = Node(4)
# d[4] = n4
# print(d[4])


def lengthOfLongestSubstring(s: str) -> int:
    left, res = 0, 0
    window_set = set()
    for i in range(len(s)):
        while s[i] in window_set:
            window_set.remove(s[left])
            left += 1
        window_set.add(s[i])
        res = max(res, i - left + 1)
    return res
# test = ["abcabcbb",
# "bbbbb",
# "pwwkew"]
# for t in test:
#     print(lengthOfLongestSubstring(t))

def lengthOfLongestSubstringKDistinct(s: str, k):
    window_cnt = {}
    left, res = 0 ,0
    for i in range(len(s)):
        if s[i] not in window_cnt: window_cnt[s[i]] = 1
        else: window_cnt[s[i]] += 1
        while len(window_cnt) > k:
            window_cnt[s[left]] -= 1
            if window_cnt[s[left]] == 0:
                del window_cnt[s[left]]
            left += 1
        res = max(res, i - left + 1)
    return res
# print(lengthOfLongestSubstringKDistinct('eeeeceba', 3))

from collections import Counter
def minWindow(s: str, t: str) -> str:
    res_start, res_end = -1, -1
    left, min_len = 0, float('inf')
    tar = dict(Counter(t))
    seen = dict()

    def valid(d, tar):
        for k, v in tar.items():
            if k not in d:
                return False
            if d[k] < v:
                ret = False
        return ret

    for i in range(len(s)):
        if s[i] in tar:
            if s[i] in seen:
                seen[s[i]] += 1
            else:
                seen[s[i]] = 1
            if len(seen) < len(tar):
                continue
            while valid(seen, tar):
                if i - left + 1 < min_len:
                    min_len = i - left + 1
                    res_start = left
                    res_end = i
                if s[left] in seen:
                    seen[s[left]] -= 1
                    if seen[s[left]] == 0:
                        del seen[s[left]]
                # print(left, i, seen)

                left += 1
    if res_end == -1:
        return ''
    print(seen)
    return s[res_start: res_end + 1]
print(minWindow("a", 'aa'))

# def is_greater(d1, d2):
#     for v, k in d1

def longestSubstring(s: str, k: int) -> int:
    # there're 26 possible unique letter in such substring
    # finding longestSubstring is to find the max len over the 26 possible choice
    res = 0
    def valid(d, uniq, k):
        # d should contain exactly uniq keys
        # and each value of it should be > k
        return len(d) == uniq and all([i >= k for i in d.values()])
    for unique_num in range(1, 27):
        # find the longest substring with exactly unique_num number of unique letter
        # shifting letter
        left = 0
        letter_cnt = dict()
        for i in range(len(s)):
            # add this letter into the dict
            if s[i] in letter_cnt: letter_cnt[s[i]] += 1
            else: letter_cnt[s[i]] = 1
            # taking too many uniq letters, trim left till valid
            while len(letter_cnt) > unique_num:
                letter_cnt[s[left]] -= 1
                if letter_cnt[s[left]] == 0:
                    del letter_cnt[s[left]]
                left += 1
            if valid(letter_cnt, unique_num, k):
                res = max(res, i - left + 1)
    return res
# print(longestSubstring("aacbbbdc", 2))


def countGoodRectangles(rectangles) -> int:
    max_side, cnt = 0, 0
    for l, w in rectangles:
        side = min(l, w)
        if max_side < side:
            max_side = side
            cnt = 1
        elif max_side == side:
            cnt += 1
    return cnt
# print(countGoodRectangles([[5,8],[3,9],[5,12],[16,5]]))


def tupleSameProduct(nums) -> int:
    res, n = 0, len(nums)
    set_nums = set(nums)
    def valid_tuple(t):
        # print(t)
        this = set(t)
        a,b,c = t
        retval = 0
        if a * b / c not in this and a * b / c in set_nums:
            this.add(a * b // c)
            retval += 1
        if a * c / b not in this and a * c / b in set_nums:
            this.add(a * c // b)
            retval += 1
        if b * c / a not in this and b * c / a in set_nums:
            this.add(c * b // a)
            retval += 1
        print(t, retval, this)
        return retval
    def bt(start, tmp):
        nonlocal res
        if len(tmp) == 3:
            res += valid_tuple(tmp)
            return
        for i in range(start, n):
            tmp.append(nums[i])
            bt(i+1, tmp)
            tmp.pop()
    bt(0, [])
    return res * 8

from collections import defaultdict
def tupleSameProduct(nums) -> int:
    d = defaultdict(list)
    res = 0
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            ratio = nums[i] / nums[j]
            if ratio in d:
                for s in d[ratio]:
                    if len(s.union(set([nums[i], nums[j]]))) == 4:
                        print(s, [nums[i], nums[j]])
                        res += 1
            d[ratio].append(set([nums[i],nums[j]]))
    # print(d)
    return (res) * 4

# print(tupleSameProduct([1,2,4,5,10]))

def subarraysWithKDistinct(A, K: int) -> int:
    # subarrayWithAtMostKDistinct(A, k)
    def subarrayWithAtMostKDistinct(A, k):
        left, res = 0, 0
        cnt = {}
        for i in range(len(A)):
            c = A[i]
            if c in cnt:
                cnt[c] += 1
            else:
                cnt[c] = 1
            while len(cnt) > k:
                cnt[A[left]] -= 1
                if cnt[A[left]] == 0:
                    del cnt[A[left]]
                left += 1
            res += i - left + 1
        return res
    return subarrayWithAtMostKDistinct(A, K) - subarrayWithAtMostKDistinct(A, K-1)

from functools import lru_cache
def numDecodings(s: str) -> int:
    if not s or len(s) == 0 or s[0] == '0': return 0
    dp = [0] * len(s)
    dp[0] = 1
    for i in range(1, len(s)):
        if 10 <= int(s[i-1:i+1]) <= 26:
            dp[i] += dp[i-2] if i - 2 >= 0 else 1
        if s[i] != '0':
            dp[i] += dp[i-1]
    print(dp)
    return dp[-1]

# print(numDecodings('226'))

def wordBreak(s, wordDict) -> bool:
    d = set(wordDict)
    n = len(s)
    dp = [[] for _ in range(n+1)]
    for i in range(n+1):
        for j in range(i):
            if (j == 0 or len(dp[j]) > 0) and s[j:i] in d:
                dp[i].append(j)
    for i in dp[-1]:
        end = n
        start = i
        while start != 0:
            print(s[start: n])
            end = start
            start = dp[start]

# print(wordBreak("leetcode", ["leet", "code", "le", 'etc', 'ode']))


def maxProduct(nums) -> int:
    n = len(nums)
    dp_max = [0] * (n + 1)
    dp_min = [0] * (n + 1)
    dp_max[0] = 1
    dp_min[0] = 0
    for i in range(len(nums)):
        # check i - 1
        if nums[i] > 0:
            dp_max[i+1] = max(dp_max[i] * nums[i], nums[i])
        elif nums[i] < 0:
            dp_max[i+1] = dp_min[i] * nums[i]
            dp_min[i+1] = min(dp_max[i] * nums[i], nums[i])
    print(dp_max, dp_min)
    return max(dp_max[1:])
# print(maxProduct([-2, 0, -1]))


def rob(nums) -> int:
    n = len(nums)
    rob_yes = [0] * n
    rob_yes[-1] = nums[-1]
    rob_no = [0] * n
    for i in range(n - 1)[::-1]:
        # dp[i, yes] = max(dp[i+1, no] + A[i], dp[i+1, yes])
        # dp[i, no] = dp[i+1, yes]
        rob_yes[i] = max(rob_no[i + 1] + nums[i], rob_yes[i + 1])
        rob_no[i] = rob_yes[i + 1]
    print(rob_no, rob_yes)
    return max(rob_no[0], rob_yes[0])
def maxP(prices):
    n = len(prices)
    dp = [[0] * 2 for _ in range( n)]
    dp[0][0] = 0
    dp[0][1] = -prices[0]
    for i in range(1, n):
        dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
        dp[i][1] = max(dp[i-1][1], -prices[i])  # -prices[i] since only one tx is allowed
        # dp[i][0] -prices[i] (don't have yesterday, buying in today) for unlimited amount of tx
    return dp[-1][0]  # sell them all before end of the game
# print(maxP([2,7,9,3,1]))

from bisect import bisect_left
def lengthOfLIS(nums) -> int:
    # n = len(nums)
    # dp = [1] * n
    # for i in range(1, n):
    #     for j in range(i):
    #         if nums[i] > nums[j]:
    #             dp[i] = max(dp[i], 1 + dp[j])
    # return max(dp)
    lis = []
    for i in range(len(nums)):
        idx = bisect_left(lis, nums[i])
        if idx == len(lis):
            lis.append(nums[i])
        if lis[idx] > nums[i]:
            lis[idx] = nums[i]
    return len(lis)


# print(lengthOfLIS([2,7,9,3,1]))

from collections import deque
def nextGreaterElement(nums1, nums2):
    res2 = {}
    stack = deque([])
    for i in range(len(nums2)):
        while stack and stack[-1][0] < nums2[i]:
            tmp = stack.pop()
            res2[tmp[0]] = i - tmp[1]
        stack.append((nums2[i], i))
    return [res2[t] if t in res2 else -1 for t in nums1]


def maxArea(height) -> int:
    # double pointer, move the shorter bar
    n = len(height)
    res = 0
    left, right = 0, n - 1
    while left < right:
        area = (right - left) * min(height[left], height[right])
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
        res = max(res, area)
    return res
def isValid(s: str) -> bool:
    stack = []
    for i in s:
        if i in ('(', '{', '['):
            stack.append(i)
        elif i == ')' and stack and stack.pop(-1) != '(':
            return False
        elif i == ']' and stack and stack.pop(-1) != '[':
            return False
        elif i == '}' and stack and stack.pop(-1) != '{':
            return False
        else:
            return False
    return not stack
isValid('()')
def generateParenthesis(n: int):
    for i in range(int(10e10)):
        print(i)
    def bt(tmp, lp, rp):
        if len(tmp) == n*2:
            print(tmp)
            return tmp
        if lp < n:
            bt(tmp + '(', lp+1, rp)
        if rp < lp:
            bt(tmp + ')', lp, rp+1)
    bt('', 0, 0)
    return
# generateParenthesis(5)


def numSquares(n: int) -> int:
    dp = [float('inf')] * n  # dp[i-1] is the number of sq's needed to get i
    s = 1
    for i in range(n):
        # the number is i+1
        if i+1 == s * s:
            dp[i] = 1
            s += 1
        else:
            for j in range(1, s+1):
                dp[i] = min(dp[i], dp[i-j*j])
                dp[i] += 1
    print(dp)
    return dp[-1]
# print(numSquares(12 ))

# 402
def removeKdigits(num, k):
    # write your code here
    stack = []
    for i in num:
        while stack and int(i) < int(stack[-1]) and k > 0:
            stack.pop()
            k -= 1
        stack.append(i)
        print(stack)
    for _ in range(k): stack.pop()
    trim = 0
    for i in stack:
        if i == '0':
            trim += 1
        else:
            break
    res = ''.join(stack[trim:])
    return res if res else '0'


from heapq import heappush, heappop, heapify

def numberofrooms(intervals):
    intervals.sort(key= lambda x: x.start)
    pq = []
    res = 0
    for i in intervals:
        if pq and pq[0][0] > i.start:
            heappush(pq, [i.end, i.start])
            res = max(res, len(pq))
        else:
            heappop(pq)
            heappush(pq, [i.end, i.start])
    return res

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
@lru_cache(None)
def maxsolitaire(a, b, c):
    if (a == 0 and b == 0) or (a == 0 and c == 0) or (b == 0 and c == 0):
        return 0
    ab = maxsolitaire(a-1, b-1, c) if a > 0 and b > 0 else float('-inf')
    bc = maxsolitaire(a, b-1, c-1) if b > 0 and c > 0 else float('-inf')
    ac = maxsolitaire(a-1, b, c-1) if a > 0 and c > 0 else float('-inf')
    return max([ab, bc, ac]) + 1
def maxsolitaire(a, b, c):
    dp = [[[0] * (c+1) for _ in range(b+1)] for _ in range(a+1)]
    for s in range(a+b+c+1):
        for i in range(a+1):
            for j in range(b+1):
                k = s - i - j
                if (i == 0 and j == 0 and 0 <= k < c) or (i == 0 and k == 0 and 0 <= j < b) or (k == 0 and j == 0 and 0 <= i < a):
                    dp[i][j][k] = 0
                elif 1 <= i <= a and 1 <= j <= b and 1 <= k <= c:
                    dp[i][j][k] = 1 + max([dp[i-1][j-1][k], dp[i][j-1][k-1], dp[i-1][j][k-1]])
    return (dp[a][b][c])

def maxs(a,b,c):
    i, j, k = sorted([a,b,c])
    res = 0
    diff = k - j
    if i > diff:
        l, s = (i + diff) // 2 , (i - diff) // 2
        res += l + s
        i -= l + s
        res += min(j - s, k - l) if j-s == k-l else min(j - s, k - l) + i
        return res
    else:
        return i + min(j, k-i)
# print(maxs(6,2,1))
# class Solution:
#     def maximumScore(self, a: int, b: int, c: int) -> int:
#         memo = {}
#         def maxScore(a: int, b: int, c: int) -> int:
#             nonlocal memo
#             if (a == 0 and b == 0) or (a == 0 and c == 0) or (b == 0 and c == 0):
#                 return 0
#             if a == 0: return min(b,c)
#             if b == 0: return min(a,c)
#             if c == 0: return min(b,a)
#             ab = (memo[(a-1,b-1,c)] if (a-1, b-1, c) in memo else maxScore(a-1, b-1, c)) if a > 0 and b > 0 else float('-inf')
#             bc = (memo[(a,b-1,c-1)] if (a, b-1, c-1) in memo else maxScore(a, b-1, c-1)) if b > 0 and c > 0 else float('-inf')
#             ac = (memo[(a-1,b,c-1)] if (a-1, b, c-1) in memo else maxScore(a-1, b, c-1)) if a > 0 and c > 0 else float('-inf')
#             memo[(a,b,c)] = max([ab, bc, ac]) + 1
#             return memo[(a,b,c)]
#         return maxScore(a,b,c)
#
# print(Solution().maximumScore(1, 4, 6))
def maxAreaOfIsland(grid):
    def dfs(x, y, dir_char, path):
        dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        dir_str = ['d', 'u', 'r', 'l']
        if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]) or grid[x][y] != 1: return ''
        grid[x][y] = 0
        path.append(dir_char)
        for i in range(len(dirs)):
            dfs(x + dirs[i][0], y + dirs[i][1], dir_str[i], path)
        return
    shapes = set()
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                path = ['#']
                dfs(i, j, '', path)
                shapes.add(''.join(path))
    return len(shapes)
# print(maxAreaOfIsland([
#     [1,1,0,0,0],
#     [1,1,0,0,0],
#     [0,0,0,1,1],
#     [0,0,0,1,1]
#   ]))

# def klarger(nums, k):
#     def divide(nums, start, end, k):
#
#
#
#     def conquer(nums, start, end):
#         # return a wall index st left is smaller and right is larger
#         wall = start
#         pivot = nums[end]
#         for i in range(start, end+1):

# quicksort
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
    return wall  # return pivot location
l = [10, 7, 8, 9, 1, 5]
# print(quickSort(l, 0, 5))

def getSkyline(buildings):
    # extension of counting planes in sky
    # image the height as number of plane in sky, the skyline is given by max height (number of plane)
    # the skyline changes when plane up/down
    pq, plane = [0], []
    for i in buildings:
        plane.append((i[0], i[2]))
        plane.append((i[1], -i[2]))
    plane.sort(key=lambda x: (x[0], x[1]))  # sorted base on time
    prev_max = 0
    res = []
    for i in plane:
        if i[1] < 0:  # down
            pq.remove(i[1])
            heapify(pq)
        else:  # up
            heappush(pq, -i[1])
        max_plane = -pq[0]
        if max_plane != prev_max:
            res.append([i[0], max_plane])
            prev_max = max_plane
    return res
# print(getSkyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]))
def hasPath(maze, start, destination):
    # write your code here
    # write your code here
    dirs = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    q = [(0, start[0], start[1])]
    # visited = [[False for _ in range(len(maze[0]))] for _ in range(len(maze))]
    visited = [[float('inf') for _ in range(len(maze[0]))] for _ in range(len(maze))]

    visited[start[0]][start[1]] = 0
    while q:
        tmp = heappop(q)
        dist = -tmp[0]
        cur = tmp[1:]
        for dir in dirs:
            x = cur[0] + dir[0]
            y = cur[1] + dir[1]
            cnt = 0
            while 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] == 0:
                x += dir[0]
                y += dir[1]
                cnt += 1
            x -= dir[0]
            y -= dir[1]
            if visited[cur[0]][cur[1]] + cnt < visited[x][y]:
                visited[x][y] = dist + cnt
                heappush(q, (-dist-cnt, x,y))

    return visited[destination[0]][destination[1]]
# print(hasPath([
#  [0,0,1,0,0],
#  [0,0,0,0,0],
#  [0,0,0,1,0],
#  [1,1,0,1,1],
#  [0,0,0,0,0]
# ], [0,4], [4,4]))

def subsets(nums):
    # DP method
    # dp(i) = dp(i-1) concat [nums[i]] + tmp for tmp in dp(i-1)
    # res = [[]]
    # for i in nums:
    #     res += [j + [i] for j in res]
    # return res
    #
    # dfs method
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
# print(subsets([2,1,4]))


def partitionLabels(S: str):
    # sweeping line
    d = {}
    for i in range(len(S)):
        s = S[i]
        if s in d:
            d[s][1] = i
        else:
            d[s] = [i, -1]
    plane = list(d.values())
    plane.sort(key=lambda x: x[0])

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
def level_to_tree(l):
    n = len(l)
    root = TreeNode(l[0])
    def helper(root, i):
        if not root: return
        vleft = TreeNode(l[2*i+1]) if 2*i+1 < n else None
        vright = TreeNode(l[2*i+2]) if 2*i+2 < n else None
        if vleft:
            root.left = vleft
        if vright:
            root.right = vright
        helper(root.left, 2*i+1)
        helper(root.right, 2*i+2)
    helper(root, 0)
    return root
# tree = level_to_tree([[3,5,1,6,2,0,8,None,None,7,4]])

import collections
import heapq
def reductionOperations(nums) -> int:
    # HEAP ?
    cnt = dict(collections.Counter(nums))
    minheap = list(cnt.keys())
    action = 0
    heapq.heapify(minheap)
    # print(minheap)
    while len(minheap) >= 2:
        k = heapq.heappop(minheap)
        cnt[minheap[0]] += cnt[k]
        action += cnt[k]
    return action


# reductionOperations([1,1,2,2,3]* -1)


def maxValue(n: str, x: int) -> str:
    start = 1 if n[0] == '-' else 0
    for i in range(start, len(n)):
        c = n[i]
        if start == 1 and int(c) > x:
            print(i, int(c))
            return n[:i] + str(x) + n[i:]
        elif start == 0 and int(c) < x:
            return n[:i] + str(x) + n[i:]
    return n + str(x)

# maxValue('-123', 3)

def minPairSum(nums) -> int:
    # sort
    snums = sorted(nums)
    res = float('-inf')
    print(snums)
    n = len(snums)
    for i in range(n // 2):
        print(snums[i], snums[n-i-1])
        res = max(res, snums[i] + snums[n-i-1])
    return res
# minPairSum([3,5,4,2,4,6])



# def maximumRemovals(s: str, p: str, removable) -> int:
#     # brute force?
#     # enumerate all possible indice of subsequence
#     # longest removable with no touch
#     m = len(p)
#     n = len(s)
#     res = []
#     i2k = {v: k for k, v in enumerate(removable)}
#     @lru_cache(None)
#     def bt(start, n_tmp, x):
#         if n_tmp == m:
#             res.append(x)
#             return
#         for i in range(start, n):
#             if p[n_tmp] == s[i]:
#                 if i in i2k : x = min(x, i2k[i])
#                 bt(i + 1, n_tmp + 1, x)
#
#     bt(0, 0, float('inf'))
#     retval = max(res)
#     return retval if retval != float('inf') else len(removable)


def maximumRemovals(s: str, p: str, removable) -> int:
    def check(s, p, remove):
        i = j = 0
        ns, np = len(s), len(p)
        while i < ns and j < np:
            if i in remove:
                i += 1
                continue
            if s[i] == p[j]:
                i += 1
                j += 1
            else:
                i += 1
        return j >= np
    start, end = 0, len(removable) - 1
    while start <= end:
        mid = (start + end) // 2
        remove = set(removable[:mid+1])  # take + 1 to cover all
        if check(s, p, remove):
            start = mid + 1
        else:
            end = mid - 1
    return end

# print(maximumRemovals("abcacb","ab",[3,1,0]))
# print(maximumRemovals("qlevcvgzfpryiqlwy", "qlecfqlw", [12,5]))
def removeDuplicateLetters(s: str) -> str:
    # lex ASC stack with duplicate count
    freq = dict(collections.Counter(s))
    stack = []
    for i in range(len(s)):
        while stack and freq[stack[-1]] > 0 and s[i] < stack[-1]: stack.pop()
        stack.append(s[i])
        freq[s[i]] -= 1
    return ''.join(stack)
# print(removeDuplicateLetters("bcabc"))
def largestRectangleArea(heights) -> int:
    stack = []
    n = len(heights)
    res = 0
    for i in range(n):
        while stack and heights[i] < heights[stack[-1]]:
            top = stack.pop()
            width = i - (stack[-1] + 1 if stack else 0)
            res = max(res, heights[top] * width)
        stack.append(i)
    while stack:
        top = stack.pop()
        width = n - (stack[-1] + 1 if stack else 0)
        res = max(res, heights[top] * width)
    return res

# print(largestRectangleArea([2,4]))

def wild_match(sc, pc):
    return sc == pc or pc == '.'

def isMatch(s: str, p: str) -> bool:
    ns, np = len(s), len(p)
    dp = [[False for _ in range(np + 1)] for _ in range(ns + 1)]
    for i in range(ns+1):
        for j in range(np+1):
            if i == j == 0:
                dp[i][j] = True
                continue
            if j == 0:
                dp[i][j] = False
                continue
            if p[j-1] != '*':
                if i > 0 and wild_match(s[i-1], p[j-1]):
                    dp[i][j] = dp[i-1][j-1]
            else:
                if j >= 2:
                    dp[i][j] |= dp[i][j-2]
                if i >= 1 and j >= 2:
                    dp[i][j] |= dp[i-1][j] and wild_match(s[i-1], p[j-2])
    pprint(dp)
    return dp[ns][np]
    # dp[0][0] = True
    # for i in range(1, ns + 1):
    #     for j in range(1, np + 1):
    #         if p[j - 1] != '*':
    #             if wild_match(s[i - 1], p[j - 1]):
    #                 dp[i][j] |= dp[i - 1][j - 1]
    #         else:
    #             if j - 2 >= 0:
    #                 dp[i][j] |= dp[i][j - 2]
    #             if wild_match(s[i - 1], p[j - 2]):
    #                 dp[i][j] |= dp[i - 1][j]
    # pprint(dp)
    # return dp[ns][np]
# isMatch("mississippi", "mis*is*p*.")

def countSubIslands(grid1, grid2) -> int:
    # DFS
    dirs = [[0, 1], [1, 0], [-1, 0], [0, -1]]

    def dfs(grid, x, y, n, check=False):
        nonlocal island
        rv = True if n != -1 else False
        if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]) or grid[x][y] != 1:
            return
        stack = [(x, y)]
        while stack:
            i, j = stack.pop()
            if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != 1:
                continue
            if check:
                if (i, j) not in island or island[(i, j)] != n:
                    rv = False
            else:
                island[(i, j)] = n
            grid[i][j] = 0
            for d in dirs:
                stack.append((i + d[0], j + d[1]))
        return rv

    island = {}

    cnt = 0
    for i in range(len(grid1)):
        for j in range(len(grid1[0])):
            if grid1[i][j] == 1:
                dfs(grid1, i, j, cnt)
                cnt += 1
    # print(island)
    res = 0
    for i in range(len(grid2)):
        for j in range(len(grid2[0])):
            if grid2[i][j] == 1:
                rv = dfs(grid2, i, j, island[(i, j)] if (i, j) in island else -1, True)
                if rv:
                    res += 1
    return res
# g1 = [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]]
# g2 = [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]
# print(countSubIslands(g1, g2))
def minDifference(nums, queries):
    table = [[0] * (len(nums)+1) for _ in range(101)]
    for i in range(1, len(nums)+1):
        table[nums[i-1]][i] = 1
    for i in range(101):
        cnt = 0
        for j in range(len(nums)+1):
            cnt += table[i][j]
            table[i][j] = cnt
    print(table)

    def query_result(q):
        i, j = q
        nonlocal table
        res = float('inf')
        prev = -1
        for n in range(101):
            if table[n][i] < table[n][j+1]:
                if prev != -1:
                    res = min(res, n - prev)
                prev = n
        return res if res != float('inf') else -1

    return [query_result(q) for q in queries]
# print(minDifference([1,3,4,8],[[0,1],[1,2],[2,3],[0,3]]))


# def splitString(s: str) -> bool:
#     # brute force ?
#     n = len(s)
#     res = False
#     def bt(i, num=-1):
#         nonlocal res
#         if i >= n:
#             res = True
#             return
#         for j in range(i, n+1):
#             new = int(s[i:j+1])
#             if num == -1 or new == num - 1:
#                 bt(j+1, new)
#
#     bt(0)
#     return res

def splitString(s: str) -> bool:
    def bt(i, tmp, num):
        if ''.join(tmp) == s:
            return True
        for j in range(i, len(s)):
            if int(s[i: j+1]) == num - 1:
                tmp.append(s[i: j+1])
                if bt(j+1, tmp, num-1):
                    return True
                tmp.pop()
        return False
    for i in range(len(s) - 1):
        if bt(i+1, [s[:i+1]], int(s[:i+1])):
            return True
    return False


def getMinSwaps(num: str, k: int) -> int:
    digits = sorted(list(num))
    n = len(digits)
    res = []
    used_ = [False for _ in range(n)]
    def bt(tmp, used):
        if all(used):
            resulted = int(''.join(tmp))
            if resulted > int(num):
                res.append(resulted)
        for i in range(n):
            if used[i] or (i != 0 and digits[i-1] == digits[i] and not used[i-1]):
                continue
            used[i] = True
            tmp.append(digits[i])
            bt(tmp, used)
            tmp.pop()
            used[i] = False
    bt([], used_)
    target = sorted(res)[k-1]
    print(target)
    v2i = {v: i for i, v in enumerate(str(target))}
    print(v2i)
    rv = 0
    for i in range(n):
        if num[i] != str(target)[i]:
            rv = max(rv, abs(v2i[num[i]] - i))
    return rv

def getMinSwaps(self, num: str, k: int) -> int:
    # part 1 find that number
    # bt = TLE
    # digits = sorted(list(num))
    n = len(num)
    # res = []
    # used_ = [False for _ in range(n)]
    # def bt(tmp, used):
    #     if all(used):
    #         resulted = int(''.join(tmp))
    #         if resulted > int(num):
    #             res.append(resulted)
    #     for i in range(n):
    #         if used[i] or (i != 0 and digits[i-1] == digits[i] and not used[i-1]):
    #             continue
    #         used[i] = True
    #         tmp.append(digits[i])
    #         bt(tmp, used)
    #         tmp.pop()
    #         used[i] = False
    # bt([], used_)
    # target = str(sorted(res)[k-1])
    # target = list('0' * (n - len(target)) + target)
    num = list(num)

    def nextPermutation(nums: List[int]) -> None:
        n = len(nums)
        if n <= 1:
            return nums
        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                break
        if nums[i] >= nums[i + 1]:
            nums.reverse()
            return
        for j in range(n - 1, i, -1):
            if nums[j] > nums[i]:
                break
        nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1:] = nums[i + 1:][::-1]

    target = list(map(int, list(num)))
    for _ in range(k):
        nextPermutation(target)
    target = list(map(str, target))
    print(target)
    # part 2 swap
    nswap = 0
    i, j = 0, 0
    print(num[4])
    while i < n:
        j = i
        while target[i] != num[j]:
            print(i, j, target[i], num[j])
            j += 1
        while i < j:
            num[j - 1], num[j] = num[j], num[j - 1]
            j -= 1
            nswap += 1
        i += 1
    return nswap
# print(getMinSwaps("059", 5))

import heapq
def minMeetingRooms(intervals):
    # Write your code here
    planes = []
    cnt = 0
    res = 0
    for i in range(len(intervals)):
        planes.append((intervals[0], 1))
        planes.append((intervals[1], -1))
    for k in planes:
        cnt += k[1]
        res = max(res, cnt)
    return res

# print(minMeetingRooms([(0,30),(5,10),(15,20)]))
def minInterval(intervals, queries):
    planes = deque(sorted(intervals))
    sorted_queries = sorted(enumerate(queries), key=lambda y: y[1])
    ans = [-1] * len(queries)
    active = []
    for idx, val in sorted_queries:
        while active and active[0][1] < val:
            heapq.heappop(active)
        while planes and planes[0][0] <= val:
            x, y = planes.popleft()
            if y >= val:
                heapq.heappush(active, (y-x+1, y))
        if active:
            ans[idx] = active[0][0]
    return ans
# print(minInterval([[1,4],[2,4],[3,6],[4,4]], [2,3,4,5]))

def sumBase(n: int, k: int) -> int:
    res = []
    while n >= k:
        q = n // k
        res.append(n - q * k)
        n = q
    print(res, n)
    return sum(res) + n
# print(sumBase(42, 2))
def longestBeautifulSubstring(word: str) -> int:
    v2i = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
    hashmap = [deque([]) for _ in range(5)]
    left, res = 0, 0

    def invalid(hashmap):
        retval = float('inf')
        for i in range(4):
            for j in range(i+1, 5):
                if max(hashmap[i]) > min(hashmap[j]):
                    retval = min(retval, min(max(hashmap[i]), min(hashmap[j])))
        return -1 if retval == float('inf') else retval

    for i in range(len(word)):
        hashmap[v2i[word[i]]].append(i)
        while all(hashmap) and invalid(hashmap) != -1:
            idx = invalid(hashmap)
            while left < i and left < idx:
                print(idx, hashmap)
                left_c = word[left]
                hashmap[v2i[left_c]].popleft()
                left += 1
            if not (left < i and left < idx):
                left_c = word[left]
                hashmap[v2i[left_c]].popleft()
                left += 1
            # invalid_res = invalid(hashmap)
        if all(hashmap) and (invalid(hashmap) == -1):
            res = max(res, i - left + 1)
            # print(word[left: i+1])
            # print(hashmap)
    return res
# test = "aeiaaioaaaaeiiiiouuuooaauuaeiu"
# for i in range(len(test)):
#     print(i,test[i])
# print(longestBeautifulSubstring("aeiaaioaaaaeiiiiouuuooaauuaeiu"))
# print(longestBeautifulSubstring("eoeiooeoiuuuaiuiuauieiuaoeieooaieuuaeuuuaaaeueaieieooeueaouuiueeeauaiiueauooaeoieeouuaaaiiiiuueiaeeiieoooeaauuaouoaoeeaoieeoeoieuueieiuuoaiuoouaioueuoiiiaooiooiuaeeeeaoaeoiiouaoieoaaeieoeoouaooaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuiooeioeueuueeuouiaaaeeoaoaaoeaooeuaoioieooieaeiaauouoooiouoaieeuuouueiiaoiuieueoeaeeaeuuiiiuaeuaeuaiieuaueaoaueauieuoaeeaaiouuueiuaaieeiiieouoaiiooueuiiuuaouuoueuuiauiaaieoaeoeaoooaiauuo"))


def maxFrequency(nums, k: int) -> int:
    minheap = [-i for i in nums]
    heapq.heapify(minheap)
    res = 0
    while minheap:
        cur = heapq.heappop(minheap)
        tmp = 1
        while minheap and cur == minheap[0]:
            heapq.heappop(minheap)
            tmp += 1
        spend = k
        for j in range(len(minheap)):
            if spend < - cur + minheap[j]:
                break
            else:
                spend -= - cur + minheap[j]
                tmp += 1
        res = max(tmp, res)
    return res
# print(maxFrequency([1,4,8,13],5))
def minJumps(arr) -> int:
    # BFS BF(uni-directional BFS)
    q, res, visited = deque([0]), 0, set([0])
    n = len(arr)
    v2i_, v2i = defaultdict(list), defaultdict(list)
    for i in range(n):
        v2i_[arr[i]].append(i)
    for key in v2i_:
        for j, v in enumerate(v2i_[key]):
            print(j,v)
            if j >= 1 and j < len(v2i_[key]) - 1 and v2i_[key][j - 1] == v - 1 and v2i_[key][j + 1] == v + 1: continue
            v2i[key].append(v)
    while q:
        sz = len(q)
        for _ in range(sz):
            cur = q.popleft()
            if cur == n - 1: return res
            for nei in [cur + 1, cur - 1] + v2i[arr[cur]][::-1]:
                if 0 <= nei < n and nei != cur and nei not in visited:
                    q.append(nei)
                    if nei == n - 1: return res + 1
                    visited.add(nei)
        res += 1
    return res
# minJumps([100,-23,-23,404,100,23,23,23,3,404])

def rotateGrid(grid, k: int):
    m = len(grid)
    n = len(grid[0])
    out = [[-1] * n for _ in range(m)]
    offset = 0
    # cnt = 0
    while grid and grid[0]:
        ret = []
        if grid and grid[0]:
            for row in grid:
                ret.append(row.pop(0))
        if grid and grid[0]:
            ret += grid.pop(-1)
        if grid and grid[0]:
            for row in grid[::-1]:
                ret.append(row.pop(-1))
        if grid and grid[0]:
            ret += grid.pop(0)[::-1]
        cnt = 0
        print(ret)
        rot = k % len(ret)
        add = ret[-rot:] + ret[: len(ret) - rot]
        print(add)
        if add:
            for i in range(offset, m - offset):
                out[i][offset] = add[cnt]
                cnt += 1
        if add:
            for i in range(offset + 1, n - offset):
                out[-offset-1][i] = add[cnt]
                cnt += 1

        if add:
            for i in range(offset, m - offset - 1)[::-1]:
                out[i][-offset-1] = add[cnt]
                cnt += 1

        if add:
            for i in range(offset + 1, n - offset - 1)[::-1]:
                out[offset][i] = add[cnt]
                cnt += 1

        offset += 1

    return out
#
# retval = rotateGrid([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], 2)
# print('-----')
# for i in retval:
#     print(i)


def distanceLimitedPathsExist(n: int, edgeList, queries):
    m, q = len(edgeList), len(queries)
    dsu = DSU(n)
    # sort both edgeList and queries by limit/edge weight (later used to define active region)
    edgeList.sort(key=lambda x: x[2])
    queries_idx = [queries[i] + [i] for i in range(q)]
    print(queries_idx)
    queries_idx.sort(key=lambda x: x[2])
    res = [False] * q
    # dsu can only add/union, so we start from the small weights
    j = 0  # edgeList ptr
    for i in range(q):
        curq = queries_idx[i]
        print(curq)
        while j < m and edgeList[j][2] < curq[2]:  # when this edgeList is active given limit
            dsu.union(edgeList[j][0], edgeList[j][1])
        res[curq[3]] = dsu.find(curq[0]) == dsu.find(curq[1])
    return res


# class DSU():
#     def __init__(self, n):
#         self.parent = [i for i in range(n)]
#         self.size = [1] * n
#
#     def find(self, x):
#         if self.parent[x] != x:
#             self.parent[x] = self.find(parent[x])
#         return x
#
#     def union(self, x, y):
#         self.parent[self.find(x)] = self.find(y)
#         rootx, rooty = self.find(x), self.find(y)
#         if rootx == rooty: return
#         if self.size[rooty] <= self.size[rootx]:
#             self.parent[rooty] = rootx
#             self.size[rootx] += self.size[rooty]
#         else:
#             self.parent[rootx] = rooty
#             self.size[rooty] += self.size[rootx]

def getOrder(tasks):
    n = len(tasks)
    pending = [(tasks[i][0], tasks[i][1], i) for i in range(n)]  # sorted by starttime
    pending.sort(key=lambda x: (x[0], x[1]))
    pending = deque(pending)
    active = []   # minheap
    heapq.heappush(active, (pending[0][1], pending[0][2]))
    i = pending[0][0]  # start time

    print(pending, active)
    pending.popleft()
    res = []
    while active or pending:
        curr = heapq.heappop(active)
        i += curr[0]
        res.append(curr[1])
        while pending and pending[0][0] <= i:
            s, t, idx = pending.popleft()
            heapq.heappush(active, (t, idx))
        if not active and pending:
            s, t, idx = pending.popleft()
            heapq.heappush(active, (t, idx))
            i = s
    return res
# print(getOrder([[5,2],[7,2],[9,4],[6,3],[5,10],[1,1]]))
import math
def getXORSum(arr1, arr2) -> int:
    if len(arr1) == 1 and len(arr2) == 1: return arr1[0] & arr2[0]
    if len(arr1) == 1:
        single = arr1[0]
        return reduce(xor, [single & i for i in arr2])
    if len(arr2) == 1:
        single = arr2[0]
        return reduce(xor, [single & i for i in arr1])
    max_ = max(max(arr1), max(arr2)) + 1
    digit = math.ceil(math.log2(max_))
    res = 0
    for i in range(digit, -1, -1):

        res *= 2
        if sum([(1 << i) & n1 for n1 in arr1]) * sum([(1 << i) & n2 for n2 in arr2]) % 2 == 1:
            res += 1
    return res
# t1 = [818492001,823729238,2261353,747144854,478230859,285970256,774747711,860954509,245631564,634746160]
# t2= [967900366,340837476]
# print(getXORSum(t1, t2))
from collections import OrderedDict

def findTheWinner(n: int, k: int) -> int:
    alive = [i for i in range(1, n + 1)]
    while len(alive) > 1:
        dead = (k % len(alive) - 1) % len(alive)
        alive = alive[dead:] + alive[:dead]
        alive.pop(0)
    return alive[0]

# print(findTheWinner(6, 5))

def minSideJumps(obstacles) -> int:
    # greedy
    # jump to farthest point at any lane
    n = len(obstacles)
    reachable = [[] for _ in range(3)]
    for i in reachable:
        i.append(([0, -1]))
    for i, lane in enumerate(obstacles):
        if lane == 0:
            continue
        if reachable[lane-1][-1][0] == i:
            reachable[lane - 1][-1][0] = i+1
            continue
        reachable[lane-1][-1][1] = i-1
        reachable[lane-1].append([i+1, -1])
        for j in range(3):
            if j != lane - 1:
                reachable[j][-1][1] = max(reachable[j][-1][1], i)
    for i in range(3):
        reachable[i][-1][1] = n
        if len(reachable[i]) == 1:
            del reachable[i][0]
    cur = 0
    l = 2
    cnt = 0
    pprint(reachable)
    while cur < n and reachable[l-1]:
        cnt += 1
        cur = reachable[l-1].pop(0)[1]
        max_, idx_ = -1, -1
        for i in range(3):
            if i != l-1:
                while reachable[i] and reachable[i][0][1] < cur:
                    reachable[i].pop(0)
                if not reachable[i]:
                    return cnt
                if reachable[i] and reachable[i][0][1] > max_ and reachable[i][0][0] <= cur:
                    max_ = reachable[i][0][1]
                    idx_ = i
        cur = max_
        l = idx_ + 1
    return cnt
# print(minSideJumps([0,3,3,0,3,2,2,0,0,3,0]))
import math
def eliminateMaximum( dist, speed) -> int:
    time = sorted([dist[i] / speed[i] for i in range(len(dist))])[1:]
    gun = 0
    tik = 0
    res = 1
    for i in time:
        gun += math.floor(i-0.00000001) - tik
        tik = math.floor(i-0.00000001)
        gun -= 1
        if gun < 0:
            return res
        res += 1

    return res
dist = [3,5,7,4,5]
speed =[2,3,6,3,2]
# print(eliminateMaximum(dist, speed))
def spmod(power, base, primemod):
    binpw = bin(power)[2:][::-1]
    n = len(binpw)
    remain = base % primemod
    res = 1 if binpw[0] != '1' else remain
    for i in range(1, n):
        remain = (remain ** 2) % primemod
        # print(remain)
        if binpw[i] == '1':
            res *= remain
    return res % primemod
# print(spmod(25, 20, 10e9 + 7))
nums1 = [57,42,21,28,30,25,22,12,55,3,47,18,43,29,20,44,59,9,43,7,8,5,42,53,99,34,37,88,87,62,38,68,31,3,11,61,93,34,63,27,20,48,38,5,71,100,88,54,52,15,98,59,74,26,81,38,11,44,25,69,79,81,51,85,59,84,83,99,31,47,31,23,83,70,82,79,86,31,50,17,11,100,55,15,98,11,90,16,46,89,34,33,57,53,82,34,25,70,5,1]
nums2 = [76,3,5,29,18,53,55,79,30,33,87,3,56,93,40,80,9,91,71,38,35,78,32,58,77,41,63,5,21,67,21,84,52,80,65,38,62,99,80,13,59,94,21,61,43,82,29,97,31,24,95,52,90,92,37,26,65,89,90,32,27,3,42,47,93,25,14,5,39,85,89,7,74,38,12,46,40,25,51,2,19,8,21,62,58,29,32,77,62,9,74,98,10,55,25,62,48,48,24,21]


def minAbsoluteSumDiff(nums1, nums2) -> int:
    sortednum1 = sorted(nums1)
    n = len(nums1)

    def query(target):
        pos = bisect_left(sortednum1, target)
        if pos == 0:
            return abs(sortednum1[pos] - target)
        before = sortednum1[pos - 1]
        after = sortednum1[pos]
        return min(after - target, target - before)

    max_compensation = 0
    min_ = float('inf')
    idx = 0
    res = 0
    for i in range(n):
        res += abs(nums1[i] - nums2[i])
        qi = query(nums2[i])
        if - abs(nums1[i] - nums2[i]) + qi < min_:
            max_compensation = qi
            idx = i
            min_ = - abs(nums1[i] - nums2[i]) + qi
    print(max_compensation,abs(nums1[idx] - nums2[idx]),idx)
    return res - abs(nums1[idx] - nums2[idx]) + max_compensation

# print(minAbsoluteSumDiff(nums1, nums2))
# test = [1, 11, 42, 97, 111, 121]
# for i in range(len(test)):
#     this = test[i]
#     res = []
#     revthis = int(str(this)[::-1])
#     for j in range(len(test)):
#         that = test[j]
#         revthat = int(str(that)[::-1])
#         res.append((this + revthat - revthis - that) if i != j else -1)
#     print(res)

def checkSubarraySum(nums, k: int) -> bool:
    hashmap = {0: 0}  # subarray length = i - hashmap[] + 1
    prefix = 0
    prev = -1
    for i, num in enumerate(nums):
        prefix += num
        find = prefix % k
        if find in hashmap:
            if i - hashmap[find] + 1 >= 2:
                print(hashmap, i, find)
                return True
        else:
            hashmap[prefix] = i
        if prev == num == 0:
            return True
        prev = num
    return False
# checkSubarraySum([1,0], 2)
parents = [3,7,-1,2,0,7,0,2]
def query(node, val, tmp=-1):
    print(node)
    if parents[node] == -1:
        return tmp
    if tmp == -1:
        tmp = parents[node]
    else:
        tmp = max(tmp, parents[node])
    return query(parents[node], val, tmp)
# print(query(1, 15))


class NumMatrix:
    def __init__(self, matrix):
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                self.prefix[i][j] = matrix[i][j]
                if i - 1 >= 0: self.prefix[i][j] += self.prefix[i - 1][j]
                if j - 1 >= 0: self.prefix[i][j] += self.prefix[i][j - 1]
                if i - 1 >= 0 and j - 1 >= 0: self.prefix[i][j] -= self.prefix[i-1][j-1]
        pprint(self.prefix)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        res = self.prefix[row2][col2] - self.prefix[row2][col1 - 1] - self.prefix[row1 - 1][col2]
        if row1 - 1 >= 0 and col1 - 1 >= 0: res += self.prefix[row1 - 1][col1 - 1]
        return res
# test = [[], [2, 1, 4, 3], [1, 1, 2, 2], [1, 2, 2, 4]]
# obj = NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]])
# a,b,c,d = [2, 1, 4, 3]
# param_1 = obj.sumRegion(a,b,c,d)
# print(param_1)


def maxCompatibilitySum(students, mentors) -> int:
    n = len(students)
    k = len(students[0])
    res = 0
    def bt(used, i, tmp):
        nonlocal res
        if i == n:
            res = max(res, tmp)
            return
        for t in range(n):
            if not used[t]:
                used[t] = True
                score = sum([students[i][s] == mentors[t][s] for s in range(k)])
                bt(used, i+1, tmp + score)
                used[t] = False
    used_ = [False] * n
    bt(used_, 0, 0)
    return res
s = [[0,1,0,0,1,0,1,0],[0,1,0,1,0,1,1,1],[0,1,1,1,0,0,0,1],[0,1,1,0,1,0,1,1],[1,0,0,0,0,0,0,0],[0,1,1,1,1,1,1,0]]
t = [[1,0,1,0,0,1,0,1],[0,1,0,0,1,0,1,0],[0,0,1,1,1,0,1,1],[0,0,1,0,0,0,1,0],[0,0,0,1,0,0,1,0],[0,0,0,1,1,1,1,1]]
# print(maxCompatibilitySum(s, t))

# def findDifferentBinaryString(nums) -> str:
#     nums = set(nums)
#     print(n)
#     print(nums)
#     for i in range(2**n):
#         print(i)
#         str_i = bin(i)[2:]
#         str_i = (n - len(str_i)) * '0' + str_i
#         if str_i not in nums:
#             return str_i
#     return 'dd'
# print(findDifferentBinaryString(['00', '00']))
def longestValidParentheses(s: str) -> int:
    wrong = set()
    n = len(s)
    bal = 0
    for i in range(n):
        c = s[i]
        if c == '(':
            bal += 1
        else:
            if bal == 0:
                wrong.add(i)
            else:
                bal -= 1
    bal = 0
    print(wrong)
    for i in range(n)[::1]:
        c = s[i]
        if i in wrong:
            continue
        if c == ')':
            bal += 1
        else:
            if bal == 0:
                wrong.add(i)
            else:
                bal -= 1
    wrong = sorted(list(wrong))
    res = 0
    if not wrong:
        return len(s)
    if len(wrong) >= 1:
        for i in range(1, len(wrong)):
            res = max(res, wrong[i] - wrong[i - 1] - 1)
    res = max([res, wrong[0], len(s) - wrong[-1] - 1])
    print(wrong)
    return res


def removeInvalidParentheses(s: str):
    # Method 1: rolling state + dfs backtracking
    rml, rmr = 0, 0
    n = len(s)
    res = set()
    def dfs(rml, rmr, i, tmp, banlance):
        print(tmp)
        if rml < 0 or rmr < 0 or banlance < 0:
            return
        if i == n:
            if rml == rmr == banlance == 0:
                res.add(''.join(tmp))
            return
        c = s[i]
        if c == '(':
            dfs(rml - 1, rmr, i + 1, tmp, banlance)
            dfs(rml, rmr, i+1, tmp + [c], banlance+1)
        elif c == ')':
            dfs(rml, rmr - 1, i + 1, tmp, banlance)
            dfs(rml, rmr, i + 1, tmp + [c], banlance-1)
        else:
            dfs(rml, rmr, i + 1, tmp + [c], banlance)
    for i in s:
        if i == '(':
            rml += 1
        elif i == ')':
            if rml != 0:
                rml -= 1
            else:
                rmr += 1
    print(rml, rmr)
    dfs(rml, rmr, 0, [], 0)
    return list(res)
# print(removeInvalidParentheses("()())()"))


def candy(ratings) -> int:
    n = len(ratings)
    nums = [1] * n
    prev = 1
    for i in range(1, n):
        if ratings[i - 1] < ratings[i]:
            nums[i] = prev + 1
        prev = nums[i]
    print(nums)
    prev = nums[-1]
    for i in range(n - 1)[::-1]:
        if ratings[i + 1] < ratings[i]:
            nums[i] = max(nums[i], prev + 1)
        prev = nums[i]
    print(nums)
    return sum(nums)
# candy([1,3,4,5,2])


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
        if k == 2:
            return self.twosum(A, target)
        for i in range(0, len(A) - k + 1):
            if i > 0 and A[i] == A[i - 1]: continue
            subset = self.kSumII(A[i + 1:], k - 1, target - A[i])
            subset.append(A[i])
            res.append(subset)
        return res

    def twosum(self, nums, target):
        # assume sorted nums, allows duplicate
        # returns twosum combinations
        res = []
        s, e = 0, len(nums) - 1
        while s < e:
            if nums[s] + nums[e] == target:
                res.append([s, e])
                s += 1
                e -= 1
                while nums[s] == nums[s - 1]: s += 1
                while nums[e] == nums[e + 1]: e -= 1
            elif nums[s] + nums[e] < target:
                s += 1
            else:
                e -= 1
        return res


def reverse(s, i, j):
    while i < j:
        s[i], s[j] = s[j], s[i]
        i += 1
        j -= 1
#
# s = ['s','b','p','v','a','c']
# reverse(s, 1, 3)
# print(s)
s = " 62   nvtk0wr4f  8 qt3r! w1ph 1l ,e0d 0n 2v 7c.  n06huu2n9 s9   ui4 nsr!d7olr  q-, vqdo!btpmtmui.bb83lf g .!v9-lg 2fyoykex uy5a 8v whvu8 .y sc5 -0n4 zo pfgju 5u 4 3x,3!wl  fv4   s  aig cf j1 a i  8m5o1  !u n!.1tz87d3 .9    n a3  .xb1p9f  b1i a j8s2 cugf l494cx1! hisceovf3 8d93 sg 4r.f1z9w   4- cb r97jo hln3s h2 o .  8dx08as7l!mcmc isa49afk i1 fk,s e !1 ln rt2vhu 4ks4zq c w  o- 6  5!.n8ten0 6mk 2k2y3e335,yj  h p3 5 -0  5g1c  tr49, ,qp9 -v p  7p4v110926wwr h x wklq u zo 16. !8  u63n0c l3 yckifu 1cgz t.i   lh w xa l,jt   hpi ng-gvtk8 9 j u9qfcd!2  kyu42v dmv.cst6i5fo rxhw4wvp2 1 okc8!  z aribcam0  cp-zp,!e x  agj-gb3 !om3934 k vnuo056h g7 t-6j! 8w8fncebuj-lq    inzqhw v39,  f e 9. 50 , ru3r  mbuab  6  wz dw79.av2xp . gbmy gc s6pi pra4fo9fwq k   j-ppy -3vpf   o k4hy3 -!..5s ,2 k5 j p38dtd   !i   b!fgj,nx qgif "
def countValidWords(sentence: str) -> int:
    tokens = sentence.split()
    res = 0
    for t in tokens:
        counter = Counter(t)
        valid = True
        hasP = counter['.'] + counter[','] + counter['!']
        if hasP > 1:
            valid = False
        for p in ('.', ',', '!'):
            if counter[p] > 1 or (counter[p] == 1 and (t[-1] != p and len(t) > 1)):
                valid = False
        if counter['-'] > 1 or (counter['-'] == 1 and (t[0] == '-' or t[len(t) - 1 - hasP] == '-')):
            valid = False
        for d in range(10):
            if str(d) in counter:
                valid = False
        res += valid
        if valid:
            print(t)
    return res
# countValidWords(s)


def l2n(l):
    res = 0
    for i in l:
        res = res * 10 + i
    return res
res = []
def construct(i, n, tmp, d, larger):
    global res
    if i == len(n):
        if l2n(tmp) > l2n(n):
            res.append(l2n(tmp))
        print(tmp)
        return
    if larger:
        rest = []
        for k in d:
            rest += [k] * d[k]
        res.append(l2n(tmp + sorted(rest)))
        return
    for j in list(d.keys()):
        if j >= n[i]:
            if d[j] == 1:
                del d[j]
            else:
                d[j] -= 1
            construct(i + 1, n, tmp + [j], d, j > n[i])
            if j in d:
                d[j] += 1
            else:
                d[j] = 1
# n = 188
#
# dic = {ii:ii for ii in [1,2]}
# construct(0, [int(d) for d in str(n)], [], dic, False)
# print(res)

def generatePalindromes(s: str):
    cnt = Counter(s)
    single = None
    half = []
    for n, c in dict(cnt).items():
        print(n, c)
        if n % 2 != 0:
            if single == None:
                single = c
            else:
                return []
        for _ in range(n // 2):
            half.append(c)
    res = []
    used = [False for _ in range(len(half))]
    def permutation(tmp, used):
        if len(tmp) == len(half):
            res.append(tmp)
            return
        for i in range(len(half)):
            if used[i] or (i > 0 and half[i-1] == half[i] and not used[half[i-1]]):
                continue
            used[i] = True
            permutation(tmp + half[i], used)
            used[i] = False
    permutation("", used)
    print(res)
    return [l + single + l[::-1] for l in res]
# print(generatePalindromes("aabb"))


def minimumOperations(nums, start: int, goal: int) -> int:
    d = {start: 0}
    goals = set()
    if 0 <= goal <= 1000:
        goals.add(goal)
    else:
        d[start] += 1
        for i in nums:
            for g in (goal - i, goal + i, goal ^ i):
                if g == start:
                    return 1
                if 0 <= g < 1000:
                    goals.add(g)
    changed = True
    while changed:
        changed = False
        for k in list(d.keys()):
            for i in nums:
                for val in (k + i, k - i, k ^ i):
                    if not 0 <= val <= 1000:
                        continue
                    if val in goals:
                        print(d)

                        return d[k] + 1
                    if val not in d:
                        d[val] = d[k] + 1
                        changed = True
    return -1
# print(minimumOperations([-290793236,101269073,-273222060,787384172,1540079,-788525404,845343284,291782798,-88261776,487025810,-586999135,-192431600,-192973099,-815073255,519371585,-402733183,792503537,397572288,-482209453,-714459341,227190468,442979184,-753042035,575402002,793575001,-950312874,922249740,-421816080,864417631,336160160,956305976,-323855560,501679215,-247459474,-172161373,-279337205,-926453531,8235430,-876293056,251172194,-765690843,-614865226,-59655013,-763664219,779242300,-861253494,-64866946,-541176714,-657503426,928811987,256882241,-662501011,358294343,-404911552,-752667723,154906373,-980027683,-782562600,552398146,610867491,-592041152,312063799,78743938,25807600,-366186767,351391659,-269237364,-95971910,516518517,420422560,-147831978,951017195,-22313395,-136791084,712100013,794095786,-369347548,499461768,-112838614,-758326139,-364941927,-606012361,-704718529,98540914,36016020,-173167948,-606657630,152848086,-220363949,534405633,-846021326,-289376430,77116849,-141427522,-830201795,534792827,943109774,-440771571,-453780530,371817517,-67496247,-864149876,42734218,713768817,-770212116,133745725,-53632661,196136539,-706202917,213788212,-839426533,687409962,-893149618,584554189,608167266,365326582,689944132,552697979,-453890018,944271565,-871909596,84744105,-359447861,508244887,507747730,210641989,728266008,230302846,214967574,-254353269,547097212,42678507,-68186188,350608118,709039652,-870778110,639496750,-701329373,-720516050,-172584948,523956714,-770590392,-725273405,202730389,461610146,201569159,-614253183,5323988,841304907,-43633189,480063220,-458178969,268344049,-876466127,682610139,-232203783,74539110,684415819,-299059184,109021730,24561356,264080297,-817788222,771519606,180166599,-135060036,-619767863,-305046836,202972661,-750598450,-757554782,227764047,621374128,-65996986,-659032432,-572590412,937060979,-471001658,-648885029,-263268751,-349084310,289517464,268475970,551585369,-775994447,541704985,-280049197,159473456,-83080863,-395673595,-566281668,-54265465,-782148722,-66032134,-657469928,282175006,353660925,-693866268,545662354,986267106,464861358,-505934790,-424501699,631253841,-434003698,-525123915,-296574812,956713309,10793850,-866005013,-634233435,-275836826,587692699,-715454224,44943545,848430461,-111422216,-921414423,657299018,-764773612,-902190745,334499666,-967255582,-734066357,-157242403,-984130642,-23787732,-989804709,834698927,861049410,248745589,495570125,898812688,-414480504,447225427,911568377,83143686,-108704236,572698561,947534886,-82828930,-432088594,981116858,186093539,-534701340,-143586341,-561599384,701429481,-75842367,987525715,-373625806,565761202,-499512059,-231163351,1]
# ,539
# ,909))

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class List:
    def __init__(self, l):
        dummy = ListNode()
        curr = dummy
        for i in l:
            curr.next = ListNode(i)
            curr = curr.next
        self.data = l
        self.head = dummy.next


# linkedlist = List([1,2,3,4]).head


def kMirror(k: int, n: int) -> int:
    def numberToBase(n, b):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return ''.join([str(i) for i in digits[::-1]])

    if n < k:
        return (n - 1) * n // 2
    n -= k
    res = (k - 1) * k // 2 + k + 1
    num = k + 2
    while n > 0:
        if str(num) == str(num)[::-1]:
            nb = numberToBase(num, k)
            if str(nb) == str(nb)[::-1]:
                res += num
                n -= 1
        num += 1
    return res
# print(kMirror(2, 5))

def getDescentPeriods(prices) -> int:
    deque = []
    n = len(prices)
    res = 0
    for i in range(n):
        if deque and deque[-1] == prices[i] + 1:
            deque.append(prices[i])
            continue
        res += len(deque) * (len(deque) - 1) // 2
        deque = [prices[i]]
    return res + n

# print(getDescentPeriods([12,11,10,9,8,7,6,5,4,3,4,3,10,9,8,7]))

from bisect import bisect_right
def LIS(nums):
    lis = []
    for i in range(len(nums)):
        idx = bisect_right(lis, nums[i])
        if idx == len(lis):
            lis.append(nums[i])
        if lis[idx] > nums[i]:
            lis[idx] = nums[i]
    print(lis)
    return len(lis)
# print(LIS([5,4,3,2,1]))
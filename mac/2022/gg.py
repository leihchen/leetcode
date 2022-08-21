from tqdm import  tqdm
from typing import *
from collections import *
import math
from bisect import bisect_left, bisect_right
import random
from random import randint
from copy import deepcopy
from heapq import heappush, heappop, heapify
from itertools import combinations
from functools import lru_cache
def longest_list(n):
    sum_ = 0
    res = []
    for i in range(1, n):
        if sum_ + i * 2 <= n:
            res.append(i*2)
            sum_ += i * 2
        else:
            break
    res[-1] += n - sum_
    return res
    # binary search

def longest_list_bs(n):
    left, right = 0, n-1
    while left <= right:
        mid = (left + right) // 2
        if n < mid * (mid + 1):
            right = mid - 1
        else:
            left = mid + 1
    res = list(range(2, 2*(right+1), 2))
    res[-1] += n - right * (right + 1)
    return res
# for i in tqdm(range(1, 100000000)):
#     assert longest_list(i * 2) == longest_list_bs(i * 2)

def valid(counter):
    max_ = max(counter)
    for c in counter:
        if c not in [0, max_]:
            return False
    return True


def longest_substr(string):
    n = len(string)
    res = ""
    for i in range(n):
        counter = [0] * 26
        for j in range(i, n):
            counter[ord(string[j]) - ord('a')] += 1
            if valid(counter):
                if sum(counter) > len(res):
                    res = string[i:j+1]
    return res

# print(longest_substr("ABCDAAABCDABCD".lower()))

# longest_substr("abcde")
# // To execute Go code, please declare a func main() in a package "main"
# // land = [0,1,2,1,2,0,1,3]
# // lava_vol =  4
# //               _
# //     _   _    |
# //   _| |X| |X X|
# // _|       |X|
# //
# //
# // land = [3,0,1,2,1,2,0,1,3]
# // lava_vol =  14
# // _               _
# //  |    _   _    |
# //  |  _| |_| |  _|
# //  |_|       |_|
#          ^
  #  0 1 2 3
  #  3,0,1 2
# 4,3,2,1,2,3,2,1,0,1,2,3,4

# scan thru the land
# while keep a array reprensent the landscape to the left
# 0,1,2
# stack.poll() ->
# stack
# 0,1,2,1,2,0,1,3
# decreasing stack
# def trapping(land):
#     stack = []  # 0,
#     res = 0
#     n = len(land)
#     for i in range(n):  # 3
#         # [3,1]
#         while stack and land[i] > land[stack[-1]]:
#             top = stack.pop()  # 2, val = 1
#             # check the left boundary of this pond
#             if not stack:
#                 break
#             left = stack[-1] # 0, val = 3
#             height = min(land[left], land[i])  # 2, 3
#             res += (height - land[top]) * (i - left - 1)  # 3-0 * (2-0-1)
#         stack.append(i)
#     return res
# print(trapping([3,0,1,2,1,2,0,1,3]))
# O(N) time
# O(N)


def longest_consistent(s):
    last = {}
    for i in range(len(s))[::-1]:
        if s[i] not in last:
            last[s[i]] = i
    start, max_len = 0, float('-inf')
    for i in range(len(s)):
        if last[s[i]] - i > max_len:
            max_len = last[s[i]] - i
            start = i
    print(s[start: last[s[start]]+1])
    return s[start: last[s[start]]+1]

# longest_consistent("cbaabaab")
# longest_consistent("performance")
# longest_consistent("adsaasss")

from collections import Counter

# https://leetcode.com/discuss/interview-question/1586927/Google-or-OA
def pair_count(nums):

    counter = Counter(nums)
    result = 0
    for sum_ in range(50 * 2 + 1):
        current_pair = 0
        for smaller in range(sum_ // 2 + 1):
            larger = sum_ - smaller
            if larger == smaller:
                current_pair += counter[larger] // 2 if larger in counter else 0
            else:
                current_pair += min(counter[larger] if larger in counter else 0, counter[smaller] if smaller in counter else 0)
        result = max(result, current_pair)
    print(result)
    return result

# pair_count([1, 9, 8, 100, 2])
# pair_count([2, 2, 2, 3])
# pair_count([5, 5])


def two_palindromes(A):
    cnt = Counter(A)
    res = 0
    visited = set()
    for k, v in cnt.items():
        if k in visited:
            continue
        if k[0] == k[1]:
            res += v * 2
            visited.add(k)
        elif k[1] + k[0] in cnt:
            res += min(v, cnt[k[1] + k[0]]) * 4
            visited.add(k[1] + k[0])
    print(res)
    return res
# https://leetcode.com/discuss/interview-question/1549789/google-interview-question-palindrome
# two_palindromes(['ck', 'kc', 'ho', 'kc'])
# two_palindromes(['ab', 'hu', 'ba', 'nn'])

# bfs https://leetcode.com/discuss/interview-question/1440227/Google-OA

from collections import defaultdict
def rod_ring(s):
    d = defaultdict(lambda: [0,0,0])
    for i in range(len(s) // 2):
        rod = int(s[2*i+1])
        color = 0 if s[2*i] == 'R' else (1 if s[2*i] == 'G' else 2)
        d[rod][color] += 1
    return sum(map(min, d.values()))

# print(rod_ring("R8R0B5G1B8G8B2R5G2R2"))
# print(rod_ring("B2R5G2R2"))



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
# print(findMaximums([0,1,2,4]))
# print(findMaximums([1,2,5,1]))
#
class Employee:
    def __init__(self, name, reporters):
        self.name = name
        self.reporters = reporters

def CEOToEmployee(employees, e):
    manager = {}
    for ee, reporters in employees:
        for reporter in reporters:
            manager[reporter] = ee
    person = e
    path = []
    while person in manager:  # while person is not ceo
        path.append(person)
        person = manager[person]
    path.append(person)
    print(path[::-1])
    return path[::-1]
# CEOToEmployee([['a', ['b']], ['b', ['c']], ['c', ['d']],], 'd')

# print(bisect_left([1,1,2,2,2,3], 2))
import collections
def f(l):
    for i in l:
        if isinstance(i, collections.Iterable) and not isinstance(i, str):
            # print('---', i)
            yield from f(i)
        else:
            yield i

i = 4
def foo(x: Iterable[int]):
    global i
    def bar():
        print(i, end='')
    # ...
    # A bunch of code here
    # ...
    for i in x:  # Ah, i *is* local to foo, so this is what bar sees
        print(i, end='')
    bar()
# foo([1,2,3])


def busiestMeetingRoom(users, k):
    frequency = Counter()
    active = [i for i in range(k)]
    heapify(active)
    used = []  # endtime, room_id
    time = 0
    for start, duration in users:
        time = max(time, start)
        while used and used[0][0] <= time:
            time, room = heappop(used)
            heappush(active, room)
        if not active:
            t, room = heappop(used)
            heappush(active, room)
            time = max(t, time)
        room = heappop(active)
        heappush(used, (start + duration, room))
        frequency[room] += 1
    max_ = 0
    res = []
    for k, v in frequency.items():
        if v == max_:
            res.append(k)
        elif v > max_:
            res = [k]
    return res
times = [1,6,10,15,16,17,17,18]
durations = [5,7,4,10,5,10,5,5]
N = 3
# print(busiestMeetingRoom(zip(times, durations), N))

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def deleteLeavesPost(root):
    res = []
    def postOrder(root):
        if not root: return
        if not root.left and not root.right:
            res.append(root.val)
            return
        postOrder(root.left)
        postOrder(root.right)
        res.append(root.val)
    postOrder(root)
    return res
root = TreeNode(val=1)
n2 = TreeNode(val=2)
n3 = TreeNode(val=3)
n4 = TreeNode(val=4)
n5 = TreeNode(val=5)
root.left = n2
root.left.left = n4
root.left.right = n5
root.right = n3
# print(deleteLeavesPost(root))


def maxChar(s):
    n = len(s)
    prev, cnt = s[0], 1
    res = 1
    for i in range(1, n):
        if s[i] == prev:
            cnt += 1
        else:
            res = max(res, cnt)
            prev = s[i]
            cnt = 1
    return res
# print(maxChar('aagggieerr'))


def max_matching(edges, n):
    prohibited = [[0] * n for _ in range(n)]
    for a, b in edges:
        prohibited[a][b] = 1
        prohibited[b][a] = 1
    for i in range(n):
        prohibited[i][i] = 1

    matches = defaultdict(lambda: -1)
    def can_play(team, visited):
        for opp in range(n):
            if prohibited[team][opp] or opp in visited:
                continue
            if matches[opp] == -1:
                matches[opp] = team
                return True
            original_opp = matches[opp]
            visited.add(opp)
            if can_play(original_opp, visited):
                matches[opp] = team
                return True

    for t in range(n):
        canPlay = can_play(t, set())
        print(matches)
    return matches





# Problem 1: Check Directions
# N, S, E, W表示东南西北, 1N2表示1在2的北边 1NW2 ‍‌‌‌‌‍‍‌‍‌‌‌‍‌‌‍‌‍‌‍表示1 在2的西南 给一个序列，检查是否合法["1N2", "3N4", "4NW5"]
# 以南北和东西分两个Topological sort解决
def nsew(directions, N):
    # aNb: a -> b, a_y > b_y
    # cWd: c -> d, c_x > d_x
    graphx = defaultdict(list)
    graphy = defaultdict(list)
    indegreex = [0] * N
    indegreey = [0] * N
    def add(a, b, d):
        if d == 'N':
            graphy[a].append(b)
            indegreey[b] += 1
        if d == 'S':
            graphy[b].append(a)
            indegreey[a] +=1
        if d == 'W':
            graphx[a].append(b)
            indegreex[b] += 1
        if d == 'E':
            graphx[b].append(a)
            indegreex[a] += 1
    def toposort(graph, indegree):
        # bfs
        t = [node for node in range(N) if indegree[node] == 0]
        q = deque(t)
        while q:
            node = q.popleft()
            for nei in graph[node]:
                indegree[nei] -= 1
                if indegree[nei] == 0:
                    q.append(nei)
                    t.append(nei)
        return len(t) == N

    for d in directions:
        a = int(d[0])
        b = int(d[-1])
        add(a, b, d[1])
        if len(d) == 4:
            add(a, b, d[2])
    if toposort(graphx, indegreex) and toposort(graphy, indegreey):
        return True
    else:
        return False

# print(nsew(["1N2", "3N4", "4NW5"], 6))
# print(nsew(["1N2", "3E1", "4W2", "4E3"], 5))
# print(nsew(['1N2', '2N3', '3N1'], 4))
# print(nsew(['1N2', '2S1'], 3))
# Problem 2: Closest Cafe for everyone
# 有一群好朋友要在学校里找个距离所有人最近的咖啡厅，前提是必须让所有的好朋友都能到达。Friends array: [1,2] Cafe array: [4,5]. 给一个二维数组代表图里面连接的边：( (1,2) , (1,5) , (2,4) ), 返回距离最近的咖啡厅。
# 以每个咖啡店为起点bfs，记录下距离，最后选取距离最短的咖啡店

# Problem 3: Restaurant waiting list
# Build a data structure to perform three operations (Restaurant is full initially):
# 1) waitList (string customer_name, int table_size):
# Add customer with given name and table size they want to book into the waitlist
# 2) leave (string customer_name):
# Customer wants to leave the waitlist so remove them.
# 3) serve (int table_size):
# This means restaurant now has a free table of size equal to table_size. Find the best customer to serve from waitlist
# .Best Customer: Customer whose required size is less than or equal to the table_size. If multiple customers are matching use first come first serve.
# For e.g. if waitlist has customers with these table requirements => [2, 3, 4, 5, 5, 7] and restaurant is serving table_size = 6 then best customer is index 3 (0-based indexing).
# TreeMap<Integer, Deque<String>> key是人数size，value是一个存储人名的队列
# HashMap<String, Integer> key是人名，value是人数size

# Problem 4: Temperature
# Given a Temperature Manager class, with functions recordTemp(int temp); and getMaxTemp(); You need to record the temperature when a new temp comes in, and get the max within the last 24 hours.
# 单调栈即可

# Problem 5: fun city
# 2242

# Problem 6: replace word by dictionary
# 给dic{"x":"123,"y":"234"}
# 然后再给一个string eg “%x%a%y%"
# return "123a234"
# follow up dic{"x":"123,"y":"%x%"}
# return become "123a123"


def formatString(d, formatingstring):

    @lru_cache(None)
    def evaluate(s: str):
        if '%' not in s:
            return s
        res = []
        i = 0
        while i < len(s):
            c = s[i]
            if c.isalnum():
                res.append(c)
            elif c == '%':
                i += 1
                j = i
                while s[i] != '%':
                    i += 1
                res.append(evaluate(d[s[j:i]]))
            i += 1
        return ''.join(res)
    return evaluate(formatingstring)
# print(formatString({"x":"123","y":"%x%"}, '%x%a%y%'))


# Problem 7: Manager Employee Management
# Design a class organization
# setManager(manager, employee) => 将employee的直属上司设为manager
# setPeer(personA, personB) => 把person A和person B设为peer，代表着这两人将有相同的上司
# reportsTo(personA, personB) => 给定两个人，返回person B是否是person A的上司，注意不一定是直属上司
# 我用的是HashMap<String, Set<String>> peers存储每个员工的peer
# HashMap<String, String> parents 存储员工的上级
# 这里注意更新一个员工的上司的时候要更新他所有的peer的上司
# class TreeNode:
#     def __init__(self, person='', direct_manager=None, direct_sub=[]):
#         self.val = person
#         self.parent = direct_manager
#         self.children = direct_sub
#
# class Org:
#     def __init__(self):
#         self.manager2sub = defaultdict(list) # HashMap<String, List<String>>
#         self.
#
#     def setManager(self, manager: str, employee: str):
#         manager_node = self.str2node[manager]
#         employee_node = self.str2node[employee]
#         manager_node.children.append(employee_node)
#         employee_node.parent = manager_node
#
#     def setPeer(self, personA, personB):
#         personA_node = self.str2node[personA]
#         personB_node = self.str2node[personB]
#         if personA_node.parent == None and
#
#     def reportsTo(self, personA, personB):

# Problem 8: Given some bigrams‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌, and with an input word, try to predict the best next word (criteria: frequency)
# [[“I”, “am”, “Sam”]
# [“Sam”, “am”]
# [“I”, “am”, “a”, “Sam”, “am”, “I”, “like”]]
# 记录frequency就行

# Problem 9: Cycle Group
# Given a list of  circles(x, y, radius), determine whether they belong to the same group. Two circles are in the same group if they overlap: dist <= r1 + r2.
# 我是用unionFind做的，for(int i=1...){for(int j=0;j<i...)}建图最后看有几个连通分量



# Tarjan's SCC algo



def validTree(n: int, edges: List[List[int]]) -> bool:
    graph = defaultdict(list)
    visited = set()
    def isCyc(node, parent):
        visited.add(node)
        for nei in graph[node]:
            if nei in visited:
                if nei != parent:
                    return True
            else:
                if isCyc(nei, node):
                    return True
        return False
    for node in range(n):
        if isCyc(node, -1):
            return False
    return True

# 有一个input matrix 每一个值表示由红灯转为绿灯的秒数 问从matrix的左上角出发到右下角需要的最少时间是多少
# 有一些assumption：一旦转换为绿灯，就不会再变红；汽车在信号灯之间的形式速度非常快，这一部分时间可以忽略不计
# Input:
# 2 3 0 5
# 1 4 5 5
# 9 8 2 7
# 6 5 4 3
# Output: 5
def trafficLight(grid):
    # bfs
    m, n = len(grid), len(grid[0])
    timetoreach = [[float('inf')] * n for _ in range(m)]
    timetoreach[0][0] = grid[0][0]
    pq = [(grid[0][0], 0, 0)]
    visited = set()
    def getNext(i, j):
        for dx, dy in [(0,1), (1,0), (-1,0), (0, -1)]:
            x, y = dx + i, dy + j
            if 0 <= x < m and 0 <= y < n:
                yield x, y, grid[x][y]
    while pq:
        cost, i, j = heappop(pq)
        if (i, j) in visited:
            continue
        visited.add((i,j))
        if i == m - 1 and j == n - 1:
            break
        for x, y, w in getNext(i, j):
            if (x, y) not in visited and max(w, cost) < timetoreach[x][y]:
                timetoreach[x][y] = max(w, cost)
                heappush(pq, (max(w, cost), x, y))
    return timetoreach[-1][-1]

# t = [[2, 3, 0, 5],
# [1, 4, 5, 5],
# [9, 8, 2, 7],
# [6, 5, 4, 3]]
# print(trafficLight(t))

def removeMinAction(nums, toRemove):
    i = 0
    while i < len(nums):
        if nums[i] in toRemove:
            nums[i], nums[-1] = nums[-1], nums[i]
            nums.pop()
        else:
            i += 1
    return nums
# print(removeMinAction([1,2,3,4,5,6], [2,4,6]))


# 设计一个NumContainer Class，用于保存一系列数字，其中包括两个方程：
#     InsertOrReplace(Index, Number)，就是在given index 的位置插入或替换数字
#     findMinimumIndex(Number)，given number出现的第一个index

from sortedcontainers import SortedList
class MyContainer:
    # HashMap + BST
    # lc 2034
    def __init__(self):
        self.index2number = {}  # <Int, Int>
        self.number2indices = defaultdict(SortedList)  # <Int, bst>

    def findMinimumIndex(self, number):
        if not self.number2indices[number]:
            raise ValueError
        print(self.number2indices[number][0])
        return self.number2indices[number][0]

    def insertOrReplace(self, index, number):
        if index in self.index2number:
            old_number = self.index2number[index]
            self.number2indices[old_number].remove(index)
        self.number2indices[number].add(index)
        self.index2number[index] = number

# myc = MyContainer()
# myc.insertOrReplace(0, 1)
# myc.insertOrReplace(1, 1)
# myc.insertOrReplace(2, 1)
# myc.insertOrReplace(3, 1)
# myc.findMinimumIndex(1)
# myc.insertOrReplace(0, 2)
# myc.insertOrReplace(2, 3)
# myc.findMinimumIndex(1)
# myc.findMinimumIndex(2)
# myc.findMinimumIndex(3)
# myc.findMinimumIndex(4)

def meetingRoomIII(intervals, rooms, asks):
    time = [0] * 500001
    for interval in intervals:
        time[interval[0]] += 1
        time[interval[1]] -= 1

    last = time[0]
    available = [0] * 11
    available[0] = 1 if last < rooms else 0
    for i in range(1, len(available)):
        curr = last + time[i]
        if curr < rooms:
            available[i] = available[i - 1] + 1
        else:
            available[i] = available[i - 1]
        last = curr
    print(available)
meetingRoomIII([[1,2],[4,5],[8,10]],
1,
[[2,3],[3,4]])

from itertools import accumulate
def meeting_room_i_i_i(intervals: List[List[int]], rooms: int, ask: List[List[int]]) -> List[bool]:
    # Write your code here.
    time = [0] * 50001
    for s, e in intervals:
        time[s] += 1
        time[e] -= 1
    presum = accumulate(time)
    avail = [cnt < rooms for cnt in presum]
    preavail = accumulate(avail)
    res = []
    for s, e in ask:
        res.append(preavail[e - 1] - preavail[s - 1] >= e - s)
    return res

# points = [(0,1), (0,-1),(1,0), (1,1), (1,2), (1,-1),(1,-2), (2,1), (2,2), (2,-1),(-1,0),(-1,1),(-1,2),(-1,-1),(-1,-2),(-2,1),(-2,-1)]
# def countSquare(points):
#     res = 0
#     seen = set()
#     for x1,y1 in points:
#         seen.add((x1,y1))
#         for x2,y2 in points:
#             if x1 != x2 and y1 != y2 and abs(x1-x2) == abs(y1-y2):
#                 if (x2, y1) in seen and (x1, y2) in seen:
#                     res += 1
#     return res

# print(countSquare(points))


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
                num = num * 10 + int(c)
            if c == '(':
                num = self.calculate(s)
            if not c.isdigit() or self.i == len(s):
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
# Solution().calculate("(4+5)-3")

def longestWordDictAnywhere(words):
    # graph on the run
    # space O(N) and time O(N)
    len2wordset = defaultdict(set)
    for w in words:
        len2wordset[len(w)].add(w)
    graph = defaultdict(list)
    indegree = Counter()
    for word in words:
        new_word = word[:-1]
        if new_word in len2wordset[len(new_word)]:
            graph[new_word].append(word)
            indegree[word] += 1
    q = deque([(chr(ord('a') + i), 1) for i in range(26)])
    parent = {}
    longest = 0
    candidate = []
    while q:
        node, level = q.popleft()
        if level > longest:
            longest = level
            candidate.clear()
        if level == longest:
            candidate.append(node)
        for nei in graph[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                parent[nei] = node
                q.append((nei, level + 1))


    res = []
    # print(graph)
    # print(candidate)
    for c in candidate:
        tmp = []
        while c in parent:
            tmp.append(c)
            c = parent[c]
        res.append(tmp[:] + [c])
    print(res)
    return res

# longestWordDictAnywhere(["o", "or", "ord", "word", "world"])
# longestWordDictAnywhere(["o", "or", "ord", "word", "world", "p", "ap", "ape", "appe", "apple"])



# 给input一堆time blocks （person_id， start_day, end_day），代表一个人在这个time block里unavailable。
# 第一问求有哪些天是所有人都available的， return list of days。
# 第二问，现在不用所有人了，只要有K个人available就行，return list of days for K people。
# TODO 第三问，现在要相同的K个人，在连续的d天里都available，return list of day ranges for a same K people。
def kPeopleFreeTime(schedule, k):
    actions = Counter()

    for s in schedule:
        for start, end in s:
            actions[start] -= 1
            actions[end] += 1
    free = len(schedule)
    res = []
    start = float('-inf')
    times = sorted(actions.keys())
    for t in times:
        action = actions[t]
        free += action
        if free >= k:
            start = min(start, t)
        if free < k:
            if start < t:
                res.append([start, t])
            start = float('inf')
    if start != float('inf'):
        res.append([start, t])
    if res[0][1] == times[0]:
        del res[0]
    else:
        res[0][0] = times[0]
    # print(res)

    return res

# kPeopleFreeTime([[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]], 1)
# kPeopleFreeTime([[[1,2],[5,6]],[[1,3]],[[4,10]]], 2)


def kPeopleDContFreeTime(schedule, k):
    actions = Counter()

    for s in schedule:
        for start, end in s:
            actions[start] -= 1
            actions[end] += 1
    free = len(schedule)
    res = []
    start = float('-inf')
    times = sorted(actions.keys())
    for t in times:
        action = actions[t]
        free += action
        if free >= k:
            start = min(start, t)
        if free < k:
            if start < t:
                res.append([start, t])
            start = float('inf')
    if start != float('inf'):
        res.append([start, t])
    if res[0][1] == times[0]:
        del res[0]
    else:
        res[0][0] = times[0]
    # print(res)

    return res

# kPeopleDContFreeTime([[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]], 1)
# kPeopleDContFreeTime([[[1,2],[5,6]],[[1,3]],[[4,10]]], 2)
# 3. （感觉是30岁左右的欧洲人小哥，可能是东欧）
# 这轮非常奇怪，一上来就开始讲故事。。。讲了一个很长的故事之后，题目大概是给三个quiz bucket，一个 total num of qu‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌estions，
# 和三个bucket的比例，实现不放回随机抽样生成一份考卷。
# 给的例子是
# total 13， ratio 0.5， 0.3， 0.2
# total 5， ratio 2， 1， 1
# 注意ratio并不一定add up to something。
# 这题说难也不难。。不知道是在考察什么，面试官一直给提示，让我挺慌的。。。
# 4. （国人大哥）
# 经典面经题 778，包装了一个红绿灯的皮，其实本质是一样的。binary search秒了，最后写了三份代码，讲了其他所有做法的思路和trade off，
# 还剩下十分钟聊天。。


# TODO 5月20号第三轮，印度小哥哥，感觉人也蛮好，很nice。题目是给一个类似于｛｛ab ｛c｝d｛｛e｝｝f｝}的字符串，要求输出需要最少数量的括号可以使得表达式依然有效。
# 譬如上面的例子正确的结果是：｛ab ｛c｝d｛e｝f｝。用stack可以解决哦。这个题答的小哥哥非常满意。
# 5月19号，onsite第一轮，这天状态不好，早晨起来喉咙疼破天了，头也嗡嗡嗡嗯。第一轮是个不开摄像头的家伙，so也不知道是啥人。上来就做题目。题目是给一个类
# ，类中是一个目录结构，就是有文件夹，有文件，这个类中有getDirectory方法，isDirectory方法，delete方法。现在让我设计一个方法来删除某个目录。
# 我先澄清了一下删除过程是否是文件为空的时候才能删除，回答是是的。那我很快用递归解决了这个问题，followup是问我递归会出现什么问题，
# 改为非递归的话应该怎么写，写完非递归（那个出口条件小哥提醒了下）又followup问了下时间和空间复杂度。
# 5月3号电面第二轮。华人小哥哥。人超级nice。第一问：给我一个arr，判断arr中某个数字连续出现最长的子串的长度；
# followup是：再给一个target arr，判断所有在target array中每个元素联系出现最长的子串的长度，

# t = '{{ab {c} d {{e}} f}}'
# def removeBrackets(s)
# 64, 1631

# 没见过的题，但感觉挺有趣，有点类似Tic Tac Toe找winner
# 两位玩家，桌上一共有12张卡片 - 3绿3白3黄3黑。
# 玩家每轮需要进行一次combine把两摞卡牌叠在一起，两位玩家交替操作。
# combine的两个条件：   
#     a. 如果两摞卡片高度一致，可以combine
#     b. 如果两摞卡片顶端的卡片颜色一致，可以combine
# 如果当前玩家无法进行combine操作，则另一名玩家获胜。
# Perfect play：100%的玩家获胜概率，即在这条路线‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌中不存在另一名玩家获胜的结果
# 题目需要写一个function，输入为player 1或player 2并默认此player是先手，输出为bool判断此player能否有perfect play。

# 在一个2d空间中，有一些Vampire和mirror，input是这些vampire和镜子的row, column 坐标和镜子方向（东南西北），设定是vampire在同一行或同一列面对自己的镜子中看到自己会embarrassed。 要求return 所有embarrassed vampire的坐标以及embarrassed的方向。比如如果vampire在左边的镜子中看到自己，embarrassed方向就是西边。vampire对同类来说是透明的。如果面向vampire的镜子被另‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌一面相反方向的镜子挡住vampire就不会看到自己。这题最后没写完，思路理得不太清楚，求大佬指点。

def setClock(dst, options):
    q = deque([0])
    visited = set()
    level = 0
    while q:
        for _ in range(len(q)):
            node = q.popleft()
            if node == dst:
                return level
            for option in options:
                new_node = (node + option) % 1440
                if new_node not in visited:
                    visited.add(new_node)
                    q.append(new_node)
        level += 1
    return -1

# print(setClock(7, (16, -9)))

def setClock2(dst, options):
    jump = defaultdict(lambda: float('inf'))
    jump[0] = 0
    q = deque([0])
    while q:
        pos = q.popleft()
        for option in options:
            if -1440 < pos + option < 1440 and jump[pos + option] > jump[pos] + 1:
                q.append(pos + option)
                jump[pos + option] = jump[pos] + 1
    return jump[dst]
# print(setClock2(7, (16, -9)))

# print(bisect_right([0, 1, 1, 3], 0))


class Graph:
    def __init__(self, vertices):
        self.V = vertices  # No. of vertices

        # dictionary containing adjacency List
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph[u].append((v, w))

    def shortestPath(self, s):
        q = deque([s])
        visited = {s}
        distance = [float('inf')] * self.V
        distance[s] = 0
        while q:
            u = q.popleft()
            for v, wUV in self.graph[u]:
                if distance[v] > distance[u] + wUV:
                    distance[v] = distance[u] + wUV
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        for i in range(self.V):
            print(("%d" % distance[i]) if distance[i] != float("Inf") else "Inf", end=" ")


# g = Graph(6)
# g.addEdge(0, 1, 5)
# g.addEdge(0, 2, 3)
# g.addEdge(1, 3, 6)
# g.addEdge(1, 2, 2)
# g.addEdge(2, 4, 4)
# g.addEdge(2, 5, 2)
# g.addEdge(2, 3, 7)
# g.addEdge(3, 4, -1)
# g.addEdge(4, 5, -2)
#
# source = 1
# s = 1
# print("Following are shortest distances from source %d " % s)
# g.shortestPath(s)



def intervlOverlopWithkey(d, q):
    planes = [d[qq] for qq in q]
    counter = Counter()
    k = len(planes)
    for start, end in planes:
        counter[start] += 1
        counter[end] -= 1
    start = float('-inf')
    onair = 0
    res = []
    for t, action in sorted(counter.items()): # secondary key?
        onair += action
        if onair == k:
            start = min(start, t)
        if onair < k:
            if t > start:
                res.append((start, t))
            start = float('inf')
    if start != float('inf') and t > start:
        res.append((start, t))
    return res

t = {'A': [1, 8],
'B': [3, 16],
'C': [5, 7],
}

# print(intervlOverlopWithkey(t, ['A', 'B']))
# print(intervlOverlopWithkey(t, ['A', 'C']))
# print(intervlOverlopWithkey(t, ['A', 'B', 'C']))

def tableJustification(strings, width):
    n = len(strings)
    left = 1
    right = n
    def valid(col):
        matrix = []
        for i in range(math.ceil(n / col)):
            matrix.append(strings[col * i: col * i + col])
        # print(matrix, col, len(matrix))
        widthneed = 0
        for j in range(col):
            max_ = max(len(matrix[i][j]) for i in range(len(matrix)-1))
            if j <= len(matrix[-1]) - 1:
                max_ = max(max_, len(matrix[i][j]))
            widthneed += max_
        widthneed += col
        # print(widthneed <= width)
        return widthneed <= width
    while left <= right:
        mid = (left + right) // 2
        if valid(mid):
            left = mid + 1
        else:
            right = mid - 1
    return right
# print(tableJustification(['foo','bar','a','b','cdefg','h','i','j','k', 'abcdefghich'], 14))
# print(tableJustification(['foo','bar','a','b','cdefg','h','i','j','k', 'i'], 14))

def reversedRecipe(recipes, menu):
    graph = defaultdict(list)
    raws = set()
    indegree = Counter()
    for dish, intermediate, raw in recipes:
        for ingredient in intermediate + raw:
            graph[dish].append(ingredient)
            indegree[ingredient] += 1
        raws |= set(raw)
    q = deque(menu)
    visited = set(q)
    while q:
        node = q.popleft()
        for nei in graph[node]:
            # indegree[nei] -= 1
            # if indegree[nei] == 0:
            if nei not in visited:
                q.append(nei)
                visited.add(nei)
    return visited & raws

# print(reversedRecipe([
#     ['Pizza', [], ['flour','cheese','ketchup','Suger']],
#     ['Steak', ['Bread', 'Salad'], ['Beef', 'oil']],
#     ['Bread', [], ['flour', 'Suger', 'oil']],
#     ['Salad', [], ['veg', 'egg', 'sauce']],
#     ['sandwich', [], ['flour', 'Suger', 'oil', 'Beef']]
# ], ['Pizza', 'Steak'])
# )
# 输入大概这样 (id, displayName, parent)
# folder(1,"folder1",-1)
# folder(2,"folder2",1)
# folder(4,"folder4",2)
# folder(3,"folder3",-1)
# 输出就是"folder1","folder1/folder2","folder1/folder2/folder4","folder3"

# def rpcTimeout(logs, timeout):
#     heapify(logs)
#     ended = set()
#     while logs:
#         time, taskid, action = heappop(logs)
#         if action == 's':
#             heappush(logs, [time + timeout, taskid, 't'])
#             # print(time, ended)
#         if action == 't' and taskid not in ended:
#             return time, taskid
#         if action == 'e':
#             ended.add(taskid)
#     return -1, -1

# print('-')
# print(rpcTimeout([[0,0,'s'], [1,1,'s'], [2,0,'e'], [4,1,'e'], [6,2,'s']], 3))


def rpcTimeout(logs, timeout):
    # running = {}  # taskid -> starttime
    # restime, restask = float('inf'), -1
    # for taskid, time, action in logs:
    #     if action == 's':
    #         running[taskid] = time
    #     if action == 'e':
    #         if time - running[taskid] > timeout:
    #             restime, restask = running[taskid] + timeout, taskid
    #             break
    #         del running[taskid]
    # print(restime, restask)
    # if not running:
    #     return restime, restask
    # else:
    #     pendingtime, pendingtask = min((v + timeout, k) for k, v in running.items())
    #     print(pendingtime, pendingtask)
    #     if pendingtime < restime:
    #         return pendingtime, pendingtask
    #     else:
    #         return restime, restask
    # optimal soln: hashmap + DLinkedList
    class Task:
        def __init__(self, id, endtime):
            self.id = id
            self.endtime = endtime
    running = deque([])
    id2task = {}
    for taskid, time, action in logs:
        if action == 'e':
            task = id2task[taskid]
            if task.endtime < time:
                return task.endtime, taskid
        if running:
            if running[0].endtime < time:
                return running[0].endtime.endtime, running[0].endtime.taskid
        if action == 's':
            task = Task(taskid, time + timeout)
            running.append(task)
            id2task[taskid] = task

start = 's'
end = 'e'
# print('-')
# print('rpc', rpcTimeout([(1, 0, start),(2,1,start),(1,2,end),(3,10,start), (3,20,end)], 3))

def majhong(cards):
    c = Counter(cards)
    res = []
    for i in sorted(c):
        if c[i] > 0:
            straight = True
            need = c[i]
            for j in range(3):
                if c[i + j] < need:
                    straight = False
            if straight:
                res.append( (str(i) + str(i+1) + str(i+2)) * need)
                for j in range(3):
                    c[i + j] -= need
    pairs = 0
    for i in c:
        if i % 3 == 0:
            res.append(str(i) * (i // 3) * 3)
            c[i] = 0
        elif i % 3 == 2:
            res.append(str(i) * (i // 3) * 3)
            res.append(str(i) * 2)
            c[i] = 0
            pairs += 1

    print(c)
    print(res)

    return pairs >= 1

majhong([1,1,1,2,3,4,6,6,7,7,8,8,9,9])
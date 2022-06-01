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

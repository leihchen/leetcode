from typing import *
from collections import *
import math
# todo domain
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        visited = Counter()
        for cpdomain in cpdomains:
            cnt, domain = cpdomain.split(' ')
            cnt = int(cnt)
            visited[domain] += cnt
            subdomain = domain.split('.')
            for i in range(1, len(subdomain)):
                visited['.'.join(subdomain[i:])] += cnt
        res = []
        for k, v in visited.items():
            res.append('{} {}'.format(v, k))
        return res


def longestCommonSubarray(user1, user2):
    m, n = len(user1), len(user2)
    dp = [[0] * (1+n) for _ in range(m+1)]
    max_ = 0
    result = []
    result2 = []
    for i in range(1, m+1):
        for j in range(1, n+1):
            if user1[i-1] == user2[j-1]:
                dp[i][j] = max(dp[i][j], 1 + dp[i-1][j-1])
                if dp[i][j] > max_:
                    max_ = dp[i][j]
                    result = user1[i-max_:i]  # user1[i-max_, i-1], len = max_
                    result2 = user2[j-max_:j]
    return result, result2, max_

# print(longestCommonSubarray(["3234.html", "xys.html", "7hsaa.html"],
#   ["3234.html", "sdhsfjdsh.html", "xys.html", "7hsaa.html"]))
# print(longestCommonSubarray([0, 1, 2, 3, 4], [4, 0, 1, 2, 3]))

# Each user completed 1 purchase.
completed_purchase_user_ids = [
    "3123122444", "234111110", "8321125440", "99911063"]
ad_clicks = [
  # "IP_Address,Time,Ad_Text",
  "122.121.0.1,2016-11-03 11:41:19,Buy wool coats for your pets",
  "96.3.199.11,2016-10-15 20:18:31,2017 Pet Mittens",
  "122.121.0.250,2016-11-01 06:13:13,The Best Hollywood Coats",
  "82.1.106.8,2016-11-12 23:05:14,Buy wool coats for your pets",
  "92.130.6.144,2017-01-01 03:18:55,Buy wool coats for your pets",
  "92.130.6.145,2017-01-01 03:18:55,2017 Pet Mittens",
]
all_user_ips = [
    # "User_ID,IP_Address",
    "2339985511,122.121.0.155",
    "234111110,122.121.0.1",
    "3123122444,92.130.6.145",
    "39471289472,2001:0db8:ac10:fe01:0000:0000:0000:0000",
    "8321125440,82.1.106.8",
    "99911063,92.130.6.144"
]

# Bought Clicked Ad Text
# 1 of 2  2017 Pet Mittens
# 0 of 1  The Best Hollywood Coats
# 3 of 3  Buy wool coats for your pets

# Counter() ad2click, Counter() ad2buy,
#

def adConversionRate(order, clicks, user_ip):
    ip2user = {}
    for elem in user_ip:
        user, ip = elem.split(',')
        ip2user[ip] = user
    ad2click, ad2order = Counter(), Counter()
    order_set = set(order)
    for elem in clicks:
        ip, _, text = elem.split(',')
        ad2click[text] += 1
        if ip in ip2user and ip2user[ip] in order_set:
            ad2order[text] += 1
    # print(ip2user)
    return ['{} of {}  {}'.format(ad2order.get(text, 0), ad2click[text], text) for text in ad2click]

# print(adConversionRate(completed_purchase_user_ids, ad_clicks, all_user_ips))


# todo course
def courseOverlap(courses):
    d = defaultdict(set)
    for sid, course in courses:
        d[sid].add(course)
    res = []
    dkey = list(d.keys())
    for i in range(len(d) - 1):
        for j in range(i+1, len(d)):
            res.append([dkey[i], dkey[j], list(d[dkey[i]] & d[dkey[j]])])
    return res

student_course_pairs_1 = [
  ["58", "Software Design"],
  ["58", "Linear Algebra"],
  ["94", "Art History"],
  ["94", "Operating Systems"],
  ["17", "Software Design"],
  ["58", "Mechanics"],
  ["58", "Economics"],
  ["17", "Linear Algebra"],
  ["17", "Political Science"],
  ["94", "Economics"],
  ["25", "Economics"],
]
student_course_pairs_2 = [
  ["42", "Software Design"],
  ["0", "Advanced Mechanics"],
  ["9", "Art History"],
]
# print(courseOverlap(student_course_pairs_1))
# print(courseOverlap(student_course_pairs_2))
all_courses = [
    ["Logic", "COBOL"],
    ["Data Structures", "Algorithms"],
    ["Creative Writing", "Data Structures"],
    ["Algorithms", "COBOL"],
    ["Intro to Computer Science", "Data Structures"],
    ["Logic", "Compilers"],
    ["Data Structures", "Logic"],
    ["Creative Writing", "System Administration"],
    ["Databases", "System Administration"],
    ["Creative Writing", "Databases"],
    ["Intro to Computer Science", "Graphics"],
]

def middleCourse(course):
    graph = defaultdict(list)
    indegree = Counter()
    outdegree = Counter()
    all_courses = set()
    for src, dst in course:
        graph[src].append(dst)
        indegree[dst] += 1
        outdegree[src] += 1
        all_courses.add(src)
        all_courses.add(dst)
    q = [c for c in all_courses if indegree[c] == 0]
    visited = set(q)
    q = deque(q)
    paths = defaultdict(list)
    for node in q:
        paths[node].append(node)
    while q:
        sz = len(q)
        for _ in range(sz):
            node = q.popleft()
            for nei in graph[node]:
                if nei not in visited:
                    paths[nei] = paths[node] + [nei]
                    visited.add(nei)
                    q.append(nei)
    res = set()
    for path in paths.values():
        if len(path) > 1:
            res.add(path[len(path) // 2 - 1])
    return list(res)

# print(middleCourse(all_courses))

# todo Rect
def findOneRect(board):
    m, n = len(board), len(board[0])
    res = []
    for i in range(m):
        for j in range(n):
            if board[i][j] == 0:
                res.append((i,j))
                height = 1
                width = 1
                while height + i < m and board[height + i][j] == 0:
                    height += 1
                while width + j < n and board[i][width + j] == 0:
                    width += 1
                res.append((i + height - 1, j + width - 1))
                break
            if res:
                break
    return res
# print(findOneRect([
#     [1,1,1,1,1],
#     [1,0,0,0,1],
#     [1,0,0,0,1],
#     [1,1,1,1,1],
#     [1,1,1,1,1]
# ]))

def findAllRect(board):
    m, n = len(board), len(board[0])
    res = []
    for i in range(m):
        for j in range(n):
            tmp = []
            if board[i][j] == 0:
                tmp.append((i,j))
                height = 1
                width = 1
                while height + i < m and board[height + i][j] == 0:
                    height += 1
                while width + j < n and board[i][width + j] == 0:
                    width += 1
                for h in range(height):
                    for w in range(width):
                        board[i+h][j+w] = 1
                tmp.append((i + height - 1, j + width - 1))
                res.append(tmp)
    return res
# print(findAllRect([
#     [1,1,1,1,1],
#     [1,0,0,0,1],
#     [1,1,1,1,1],
#     [1,0,0,1,1],
#     [1,1,1,0,0]
# ]))

# todo text
def wordWrap(words, maxLen):
    res = []
    cur = []
    curLen = 0
    for w in words:
        if curLen + len(w) + len(cur) > maxLen:
            res.append('-'.join(cur))
            cur, curLen = [], 0
        cur += [w]
        curLen += len(w)
    return res

lines = [ "The day began as still as the",
          "night abruptly lighted with",
          "brilliant flame" ]
# print(wordWrap("The day began as still as the night abruptly lighted with brilliant flame".split(), 6))


def fullJustify(words, maxWidth):
    res, cur, num_of_letters = [], [], 0
    for w in words:
        if num_of_letters + len(w) + len(cur) > maxWidth:
            for i in range(maxWidth - num_of_letters):
                cur[i % (len(cur)-1 or 1)] += '-'
            res.append(''.join(cur))
            cur, num_of_letters = [], 0
        cur += [w]
        num_of_letters += len(w)
    return res + ['-'.join(cur)]

# print(fullJustify(' '.join(lines).split(' '), 24))

# todo calculator
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

# todo 矩阵合法
def validMatrix(matrix):  # all row and col contains [1, n]
    n = len(matrix)
    for i in range(n):
        rowSet, colSet = set(), set()
        rowMin = colMin = float('inf')
        rowMax = colMax = float('-inf')
        for j in range(n):
            elem = matrix[i][j]
            if elem in rowSet:
                return False
            rowMax = max(rowMax, elem)
            rowMin = min(rowMin, elem)

            elem = matrix[j][i]
            if elem in colSet:
                return False
            colMax = max(colMax, elem)
            colMin = min(colMin, elem)
        if colMax != n or rowMax != n or colMin != 1 or rowMin != 1:
            return False
    return True

# print(validMatrix([
#     [1,2,3],
#     [3,1,2],
#     [2,3,1]
# ]))
from itertools import groupby
def nonogram(matrix, rows, cols):
    m, n = len(matrix), len(matrix[0])
    def checkReq(strings, req):
        for i in range(len(strings)):
            check = req[i]
            if not check: continue
            subcnt = [len(list(g)) for c, g in groupby(strings[i]) if c == '0']
            j = 0
            print(check, subcnt)
            for sb in subcnt:
                if j >= len(check):
                    break
                if sb == check[j]:
                    j += 1
            if j != len(check):
                return False
            # start = 0
            # for c in check:
            #     if start >= len(strings[i]):
            #         return False
            #     index = strings[i].find('0' * c, start)
            #     if index == -1:
            #         return False
            #     if index + c < len(strings[i]) and strings[i][index+c] == '0':
            #         return False
            #     start = index + c
        return True

    if not checkReq([''.join(map(str, row)) for row in matrix], rows):
        return False
    col_str = []
    for j in range(n):
        col_str.append(''.join(str(matrix[i][j]) for i in range(m)))
    if not checkReq(col_str, cols):
        return False
    return True
matrix1 = [
    [1,1,1,1], # []
    [0,1,1,1], # [1] -> a single run of _1_ zero (i.e.: "0")
    [0,1,0,1,0,0], # [1, 2] -> first a run of _1_ zero, then a run of _2_ zeroes
    [1,1,0,1], # [1]
    [0,0,1,1], # [2]
]

# True
rows1_1 = [[],[1],[1,2],[1],[2]]
columns1_1 = [[2,1],[1],[2],[]]
# False
rows1_2 = [[],[],[1],[1],[1,1]]
columns1_2 = [[2],[1],[2],[1]]
# print(nonogram(matrix1, rows1_1, columns1_1))
# print(nonogram(matrix1, rows1_2, columns1_2))

matrix2 = [
    [1,1],
    [0,0],
    [0,0],
    [1,0]
]
# False
rows2_1 = [[],[2],[2],[1]]
columns2_1 = [[1,1],[3]]
# print(nonogram(matrix2, rows2_1, columns2_1))

# todo Ancestor
def findNodesWithZeroOrOneParent(edges):
    hashmap = defaultdict(set)
    nodes = set()
    for parent, child in edges:
        hashmap[child].add(parent)
        nodes.add(child)
        nodes.add(parent)
    return [k for k in nodes if len(hashmap[k]) in (0,1)]

# print(findNodesWithZeroOrOneParent([[1,4], [1,5], [2,5], [3,6], [6,7]]))

def hasCommonAncestor(edges, x, y):
    # reverse the graph
    # dfs from the two nodes and store all nodes reachable in a set (which is already part of dfs)
    # intersection of the two sets has to be > 1
    p = defaultdict(list)
    for parent, child in edges:
        p[child].append(parent)
    def bfs(x):
        q = deque([x])
        visited = set()
        visited.add(x)
        while q:
            node = q.popleft()
            for nei in p[node]:
                if nei not in visited:
                    visited.add(nei)
                    q.append(nei)
        return visited
    px, py = bfs(x), bfs(y)
    return px & py
# print(hasCommonAncestor([[1,2], [3,4], [1,3], [5,4]], 2, 4))

# todo 门禁
def badgeRecords(records):
    wenter, wexit = set(), set()
    d = defaultdict(int)
    for name, action in records:
        if action == 'enter':
            if d[name] == 0:
                d[name] = 1
            else:
                wenter.add(name)
        else:
            if d[name] == 1:
                d[name] = 0
            else:
                wexit.add(name)
    for k, v in d.items():
        if v == 1:
            wenter.add(k)
    return [list(wenter), list(wexit)]

badge_records = [
  ["Martha",   "exit"],
  ["Paul",     "enter"],
  ["Martha",   "enter"],
  ["Martha",   "exit"],
  ["Jennifer", "enter"],
  ["Paul",     "enter"],
  ["Curtis",   "enter"],
  ["Paul",     "exit"],
  ["Martha",   "enter"],
  ["Martha",   "exit"],
  ["Jennifer", "exit"],
]

# Expected output: ["Paul", "Curtis"], ["Martha"]
# print(badgeRecords(badge_records))

def freqentAccess(records):
    res = []
    def timeDiff(a, b):
        a, b = int(a), int(b)
        aHour = math.floor(a / 100)
        bHour = math.floor(b / 100)
        aMinute = a % 100
        bMinute = b % 100
        return aHour * 60 + aMinute - (bHour * 60 + bMinute)

    name2time = defaultdict(list)
    for name, ts in records:
        name2time[name].append(ts)
    for k, v in name2time.items():
        v.sort()
        i = 0
        timewindow = [v[0]]
        for j in range(1, len(v)):
            if timeDiff(v[i], v[j]) < 60:
                timewindow.append(v[j])
            else:
                timewindow = [v[j]]
                i = j
            if len(timewindow) >= 3:
                res.append([k, timewindow])
                break
    return res
# print(freqentAccess([['James', '1300'], ['Martha', '1600'], ['Martha', '1620'], ['Martha', '1530'], ['Martha', '1900'], ['Martha', '1930']] ))

# todo meeting
def noOverlap(meetings, start, end):
    for s, e in meetings:
        if not (start >= e or end <= s):
            return False
    return True
# print(noOverlap([[1300, 1500], [930, 1200],[830, 845]], 1450, 1500))

def spareTime(intervals):
    intervals.sort(key=lambda x: (x[0], -x[1]))
    n = len(intervals)
    curmax = intervals[0][1]
    start = intervals[0][0]
    res = []
    for i in range(1,n):
        if intervals[i][0] > curmax:
            res.append([start, curmax])
            start, curmax = intervals[i]
        else:
            curmax = max(curmax, intervals[i][1])
    res.append([start, curmax])
    free = []
    start = 0
    for s, e in res:
        free.append([start, s])
        start = e
    return free

# print(spareTime([[1,3],[2,6],[8,10],[15,18]]))


# todo sparsevector
class SparseVector:
    def __init__(self, n):
        self.n = n
        self.data = defaultdict(int)

    def get(self, idx):
        if idx >= self.n:
            raise IndexError
        return self.data[idx]

    def set(self, idx, val):
        if idx >= self.n:
            raise IndexError
        self.data[idx] = val

    def dot(self, other):
        value = 0
        for key in self.data:
            value += self.data[key] * other.data[key]
        return value

    def add(self, other):
        value = 0
        for key in self.data:
            value += self.data[key] + other.data[key]
        return value

    def cos(self, other):
        thisNorm = sum([v**2 for v in self.data.values()])
        thatNorm = sum([v**2 for v in other.data.values()])
        return self.dot(other) / thisNorm / thatNorm

# todo treasure
def visitedAll(matrix, i, j):
    m, n = len(matrix), len(matrix[0])
    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n:
            if matrix[i][j] != 0:
                return
            matrix[i][j] = 1
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j+1)
            dfs(i, j-1)
    dfs(i, j)
    for row in matrix:
        for elem in row:
            if elem == 0:
                return False
    return True

def allTreasure(matrix, start, end):
    treasure = 0
    m, n = len(matrix), len(matrix[0])
    for row in matrix:
        for elem in row:
            if elem == 1:
                treasure += 1
    paths = []
    def dfs(x, y, path, tcnt):
        if 0 <= x < m and 0 <= y < n:
            if matrix[x][y] in (-1, 2):
                return
            path.append((x, y))
            temp = matrix[x][y]
            if temp == 1:
                tcnt += 1
            if (x,y) == end and tcnt == treasure:
                paths.append(list(path))
                path.pop()
                matrix[x][y] = temp
                return
            matrix[x][y] = 2
            dfs(x + 1, y, path, tcnt);
            dfs(x - 1, y, path, tcnt);
            dfs(x, y + 1, path, tcnt);
            dfs(x, y - 1, path, tcnt);
            matrix[x][y] = temp
            path.pop()
    dfs(start[0], start[1], [], 0)
    if not paths:
        return []
    min_len = min([len(p) for p in paths])
    # print(paths)
    for p in paths:
        if len(p) == min_len:
            return p
board3 = [
    [  1,  0,  0, 0, 0 ],
    [  0, -1, -1, 0, 0 ],
    [  0, -1,  0, 1, 0 ],
    [ -1,  0,  0, 0, 0 ],
    [  0,  1, -1, 0, 0 ],
    [  0,  0,  0, 0, 0 ],
]
# print(allTreasure(board3, (5,2), (2,0)))




#####
'''
The conflict with your students escalates, and now they are hiding multiple words in a single word grid. Return the location of each word as a list of coordinates. Letters cannot be reused across words.

grid1 = [
    ['b', 'a', 'b'],
    ['y', 't', 'a'],
    ['x', 'x', 't'],
]

words1_1 = ["by","bat"]

find_word_locations(grid1, words1_1) => 
([(0, 0), (1, 0)],
 [(0, 2), (1, 2), (2, 2)])

grid2 =[
    ['A', 'B', 'A', 'B'],
    ['B', 'A', 'B', 'A'],
    ['A', 'B', 'Y', 'B'],
    ['B', 'Y', 'A', 'A'],
    ['A', 'B', 'B', 'A'],
]
words2_1 = ['ABABY', 'ABY', 'AAA', 'ABAB', 'BABB']

([(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)],
 [(1, 1), (1, 2), (2, 2)],
 [(3, 2), (3, 3), (4, 3)],
 [(0, 2), (0, 3), (1, 3), (2, 3)],
 [(3, 0), (4, 0), (4, 1), (4, 2)])

or

([(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
 [(2, 0), (2, 1), (3, 1)],
 [(3, 2), (3, 3), (4, 3)],
 [(0, 2), (0, 3), (1, 3), (2, 3)],
 [(3, 0), (4, 0), (4, 1), (4, 2)])

or

([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
 [(2, 0), (2, 1), (3, 1)],
 [(3, 2), (3, 3), (4, 3)],
 [(0, 2), (0, 3), (1, 3), (2, 3)],
 [(3, 0), (4, 0), (4, 1), (4, 2)])

words2_2 = ['ABABA', 'ABA', 'BAB', 'BABA', 'ABYB']

([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
 [(3, 2), (4, 2), (4, 3)],
 [(0, 1), (0, 2), (1, 2)],
 [(0, 3), (1, 3), (2, 3), (3, 3)],
 [(1, 1), (2, 1), (3, 1), (4, 1)])

or

([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
 [(3, 2), (4, 2), (4, 3)],
 [(0, 1), (0, 2), (0, 3)],
 [(1, 2), (1, 3), (2, 3), (3, 3)],
 [(1, 1), (2, 1), (3, 1), (4, 1)])


Complexity analysis variables:

r = number of rows
c = number of columns
w = length of the word
'''

grid1 = [
    ['b', 'a', 'b'],
    ['y', 't', 'a'],
    ['x', 'x', 't'],
]
words1_1 = ['by', 'bat']

grid2 = [
    ['A', 'B', 'A', 'B'],
    ['B', 'A', 'B', 'A'],
    ['A', 'B', 'Y', 'B'],
    ['B', 'Y', 'A', 'A'],
    ['A', 'B', 'B', 'A'],
]
words2_1 = ['ABABY', 'ABY', 'AAA', 'ABAB', 'BABB']
words2_2 = ['ABABA', 'ABA', 'BAB', 'BABA', 'ABYB']


# set words, list[counter, word]
# anagram, string > word
# for each word in words:
# list.append(counter(word))
# Counter(string) compare to each list
#
# from collections import Counter

# def contains(a, b):  # O(1)
#     for k, v in b.items():
#         # a[k] >= v
#         if k not in a:
#             return False
#         if a[k] < v:
#             return False
#     return True

# def findWord(words, string):
#     word_counter = []
#     for word in words:  # O(W)
#         word_counter.append((Counter(word), word))  # O(26S)
#     for cnt, res in word_counter: # O(W)
#         if contains(Counter(string), cnt):
#             return res
#     return None
# O(WS)
# O(W)


# dfs with backtracking
# dfs(i,j,k,path): (i,j) standing at, k if the index of string to match,
# path: List[index] contain the res
#   if matrix[i][j] == string[k]:
#        path.append((i,j))
#        mark (i,j) as visited
#        call dfs() down, right
#        path.pop()
#        unmark  (i,j)
# def find_word_location(grid, word):
#     m, n = len(grid), len(grid[0])
#     res = None
#     w = len(word)
#
#     def dfs(i, j, k, path):  # O(w)
#         nonlocal res
#         if k == w:
#             res = list(path)
#             return
#         if not (0 <= i < m and 0 <= j < n):
#             return
#         if grid[i][j] == word[k]:
#             path.append((i, j))
#             temp = grid[i][j]
#             grid[i][j] = '#'
#             dfs(i + 1, j, k + 1, path)  #
#             dfs(i, j + 1, k + 1, path)  #
#             path.pop()
#             grid[i][j] = temp
#
#     for i in range(m):  # O(r)
#         for j in range(n):  # O(c)
#             dfs(i, j, 0, [])  # O(w)
#             if res:
#                 return res


# O(rcw)
# O(w)
from collections import defaultdict


# dfs()
# dfs for len(words) times
# bitmask [0] * len(words) matched
# in dfs, bitmask[w] = 1,
# for i in range(len(words)):
#   if bitmask[i] = 0:  not yet matched
#       start next round of dfs
# res = defualtdict(list)  # Dict<str, List[index]>
# def dfs(i, j, )

def find_word_locations(grid, words):
    res = defaultdict(list)  # Dict<str, List[index]>
    m, n = len(grid), len(grid[0])
    ans = []
    def dfs(i, j, curr_word, k, bitmask, curr_path, start=False):
        # (i,j) index standing at, k if curr_word index looking words[curr_word][k]
        # curr_path keeps the path for words[curr_word]
        print(bitmask)
        if k == len(words[curr_word]) or start:
            if not start and curr_path:
                res[words[curr_word]] = curr_path
                bitmask[curr_word] = 1
            if all(v == 1 for v in bitmask):
                ans.append({k: list(v) for k, v in res.items()})
                return
            for next_word in range(len(bitmask)):
                if bitmask[next_word] == 0:
                    for ii in range(m):
                        for jj in range(n):
                            dfs(ii, jj, next_word, 0, bitmask, [])
            return
        if not (0 <= i < m and 0 <= j < n):
            return
        print(i, j)
        print(curr_word, k)
        if grid[i][j] == words[curr_word][k]:
            curr_path.append((i, j))
            temp = grid[i][j]
            grid[i][j] = '#'
            dfs(i + 1, j, curr_word, k + 1, bitmask,  curr_path)  #
            dfs(i, j + 1, curr_word, k + 1, bitmask,  curr_path)  #
            curr_path.pop()
            grid[i][j] = temp

    dfs(0, 0, 0, 0, [0] * len(words), [], True)
    return ans


print(find_word_locations(grid1, words1_1))

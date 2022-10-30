# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import heapq
from collections import Counter, defaultdict, deque
from bisect import *
from heapq import *
from sortedcontainers import SortedList
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def numberQuery(q, diff):
    hashmap = {}
    res = []
    valid = 0
    for action in q:
        sign = action[0]
        number = int(action[1:])
        pairs = hashmap.get(number + diff, 0) + hashmap.get(number - diff, 0)
        if sign == '+':
            valid += pairs
            hashmap[number] = 1
        else:
            valid -= pairs
            del hashmap[number]
        res.append(valid)
    return res

# def nearlyRegular(matrix):

    # rowcnt, colcnt = defaultdict(set)
    # n = len(matrix)
    # def findidx(row):
    #     cnt = Counter(row)
    #     # val2idx = {val: idx }
    #     if len(cnt) > 2: return []
    #
    #     if cnt.items():

        # cand = row[0]
        # res = -1
        # for i in range(1, len(row)):
        #     if row[i] != cand:
        #         if res != -1:
        #             return None
        #         res = i
        # return res
    # for i in range(n):
    #     rowcnt.append(findidx(matrix[i]))
    # for j in range(n):
    #     tmp = []
    #     for i in range(n):
    #         tmp.append(matrix[i][j])
    #     colcnt.append(findidx(tmp))
    # ans = 0
    # for idx, val in enumerate(colcnt):
    #     if val and val > 0 and rowcnt[val] == idx:
    #         ans += 1
    #
    # return ans + colcnt.count(-1) * rowcnt.count(-1)

def light(objects, radius):
    lighted = deque([])
    res = 0  # [res - r, res + r]
    i, n = 0, len(objects)
    max_ = 0
    for j in range(objects[0], objects[-1] + 1):
        while lighted and lighted[0] < j - radius:
            lighted.popleft()
        while i < n and j - radius <= objects[i] <= j + radius:
            lighted.append(objects[i])
            i += 1
        if len(lighted) > max_:
            res = j
            max_ = len(lighted)
            print(lighted)
    return res
from typing import *
def centerText(p, width):
    res = []
    def add(words: List[str]):
        curLen, cur = 0, []
        for i in range(len(words)):
            word = words[i]
            if len(word) + len(cur) + curLen > width:
                tmp = ' '.join(cur)
                space = width - len(tmp)
                line = space // 2 * ' ' + tmp
                line += (width - len(line)) * ' '
                res.append(line)
                curLen, cur = 0, []
            curLen += len(word)
            cur.append(word)
        tmp = ' '.join(cur)
        space = width - len(tmp)
        line = space // 2 * ' ' + tmp
        line += (width - len(line)) * ' '
        res.append(line)
    for sents in p:
        add(sents)
    final = []
    for line in res:
        final.append('*' + line + '*')
    return ['*' * (width + 2)] + final + ['*' * (width + 2)]

def deckGame(deck1, deck2):
    turns = 0
    deck1 = deque(deck1)
    deck2 = deque(deck2)
    while deck1 and deck2:
        p1, p2 = deck1.popleft(), deck2.popleft()
        if p1 >= p2:
            deck1.extend([p1, p2])
        else:
            deck2.extend([p2, p1])
        turns += 1
    return turns

def magicOperation(nums):
    # greedy
    heapq.heapify(nums)
    while len(nums) >= 2:
        heapq.heappush(nums, 2 * min(heapq.heappop(nums), heapq.heappop(nums)))
    return heapq.heappop(nums)

def obstacle(query):
    # convert point to interval,
    intervals = SortedList()
    n = len(query)
    res = []
    for i in range(n):
        if query[i][0] == 1:
            obst = query[i][1]
            left, right = None, None
            insert = [obst, obst]
            idx = bisect_left(intervals, insert)
            if idx - 1 >= 0 and intervals[idx-1][1] == obst - 1:
                insert[0] = intervals[idx-1][0]
                left = intervals[idx-1]
            if idx < len(intervals) and intervals[idx][0] == obst + 1:
                insert[1] = intervals[idx][1]
                right = intervals[idx]
            if left: intervals.remove(left)
            if right: intervals.remove(right)
            intervals.add(insert)
        if query[i][0] == 2:
            x, size = query[i][1], query[i][2]
            start, end = x - size, x - 1
            idx = bisect_left(intervals, [x - size, x - 1])
            if (idx - 1 >= 0 and intervals[idx-1][1] >= x - size) or (idx < len(intervals) and intervals[idx][0] <= end):
                res.append(0)
            else: res.append(1)
    print(intervals)
    return ''.join(map(str, res))



def blackCount(blacks, rows, cols):
    blacks = set(map(tuple, blacks))
    res = [0] * 5
    for row_offset in range(rows-2+1):
        for col_offset in range(cols-2+1):
            tmp = set()
            for dx in (0, 1):
                for dy in (0, 1):
                    tmp.add((row_offset + dx, col_offset + dy))
            res[len(blacks & tmp)] += 1
    return res

# def lampswitch(lamps, points):
    # planes = []
    # for start, end in lamps:
    #     planes.append((start, 1))
    #     planes.append((end, -1))
    # pointidx = [[point, i] for i, point in enumerate(points)]
    # i = j = 0
    # pointidx.sort()
    # res = [-1] * len(points)
    # planes.sort(key=lambda x: (x[0], -x[1]))
    # cnt = 0
    # prev = 0
    # while i < len(planes):
    #     cnt += planes[i][1]
    #     while i + 1 < len(planes) and planes[i][0] == planes[i+1][0] and planes[i+1][1] > 0 and planes[i][1] > 0:
    #         cnt += planes[i][1]
    #         i += 1
    #     if prev <= pointidx[j][0] < planes[i][0]:
    #         res[pointidx[j][1]] = max(res[pointidx[j][1]], cnt)
    # return res

def lampswitch(lamps, points):
    actions = [(0,0)]
    pointsidx = [(point, i) for i, point in enumerate(points)]
    pointsidx.sort()
    print(pointsidx)
    for start, end in lamps:
        actions.extend([(start, 1), (end+0.01, -1)])
    cnt = 0
    res = [-1] * len(points)
    j = 0
    actions.sort(key=lambda x: (x[0], -x[1]))
    print(actions)
    for i, (ts, action) in enumerate(actions):
        if actions[i - 1][0] <= pointsidx[j][0] < ts:
            res[pointsidx[j][1]] = max(res[pointsidx[j][1]], cnt)
            j += 1
        cnt += action
    return res

def texteditor(ops):
    res = ""
    history = []
    ans = []
    def insert(word):
        nonlocal res
        history.append((1, word))
        res += word
    def backspace():
        nonlocal res
        if res:
            history.append((0, res[-1]))
            res = res[:-1]
    def undo():
        nonlocal res
        if history:
            isAdd, word = history.pop()
            if isAdd:
                res = res[:len(res) - len(word)]
            else:
                res += word
    for op in ops:
        verb = op
        word = ''
        if ' ' in op:
            verb, word = op.split(' ')
        if verb == 'INSERT':
            insert(word)
        if verb == 'BACKSPACE':
            backspace()
        if verb == 'UNDO':
            undo()
        ans.append(res)
    return ans
# coding: given two integer arrays, find a pair of values from both arrays such that if they are swapped, the sum of two arrays are the same


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p = [['hello', 'world'], ['How', 'areYou', 'doings'], ["Please look", 'and align', 'to center']]
    width = 16
    q = ["+4", "+5", "+2", "-4"]
    diff = 1
    # print(numberQuery(q, diff))

    matrix = [[1,1,1,1], [2,3,1,1], [1,1,1,0], [1,4,1,1]]
    # print(nearlyRegular(matrix))

    # print(light([-2, 4, 5, 6, 7], 1))
    # print(light([-5, 3, 4, 9], 5))

    # res = centerText(p, width)
    # for line in res:
    #     print(line)
    # print(deckGame([1,2], [3,1]))

    # print(magicOperation([10, 7, 3, 3, 5]))
    # print(magicOperation([2, 1, 4, 5]))
    # print(bisect_left( [0, 1, 1, 3], 2))

    # print(obstacle([[1,2], [1,5], [2,5,2], [2,6,3], [2,2,1], [2,3,2]]))
    # print(blackCount([[0,0], [0,1], [1,0]], 3, 3))
    # print(lampswitch([[1,7], [5,11], [7,9], [7,9]], [7,1,5,10]))
    # print(texteditor(['INSERT Code', 'INSERT Signal', 'BACKSPACE', 'UNDO']))
    # print(texteditor(['INSERT co', 'INSERT d', 'UNDO', 'BACKSPACE', 'BACKSPACE', 'BACKSPACE', 'UNDO', 'UNDO', 'UNDO', 'UNDO',
    #                   'INSERT ding']))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# distinct 1 2 7 7 4 3 6
# k = 3
#
def observable(nums, k):
    cnt = Counter()
    n = len(nums)
    sum_ = res = 0
    for i in range(k):
        cnt[nums[i]] += 1
        sum_ += nums[i]
    for left in range(n - k + 1):
        if len(cnt) == k:
            res = max(res, sum_)
        sum_ -= nums[left]
        cnt[nums[left]] -= 1
        if cnt[nums[left]] == 0:
            del cnt[nums[left]]
        if left + k >= n: break
        sum_ += nums[left + k]
        cnt[nums[left + k]] += 1
    return res

# print(observable([1, 2, 7, 7, 4, 3, 6], 3))

def keypad99(string):
    cnt = Counter(string)
    freq = sorted(cnt.values())[::-1]
    return sum(freq[:9]) + sum(freq[9:2*9]) * 2 + sum(freq[2*9:3*9]) * 3


# print(keypad99('abacadefghibj'))

def searchWord(searchword, resultword):
    # add to the end of search word, st resultword is a subseq of searchword
    i = j = 0
    n, m = len(searchword), len(resultword)
    while i < n and j < m:
        if searchword[i] == resultword[j]:
            i += 1
            j += 1
        else:
            i += 1
    return m - j
# print(searchWord('armaze', 'amazon'))
# print(searchWord('armazenon', 'amazon'))
# print(searchWord('amzon', 'amazon'))

def data_movement(data, movefrom, moveto):
    s = set(data)
    for i in range(len(movefrom)):
        s.remove(movefrom[i])
        s.add(moveto[i])
    return sorted(s)
# print(data_movement([1,7,6,8], [1,7,2], [2,9,5]))


# 5 **
def loginout(executions):
    user2pass = {}
    active = set()
    res = []
    for action in executions:
        log = action.split(' ')
        if len(log) == 2:
            if log[1] in active:
                active.remove(log[1])
                res.append('Logged Out Successfully')
        else:
            verb, username, password = log
            if verb == 'login':
                if password == user2pass.get(username, None):
                    res.append('Logged In Successfully')
                else:
                    res.append('Login Unsuccessful')
            else:
                if username in user2pass:
                    res.append('Username already exists')
                else:
                    user2pass[username] = password
                    res.append('Registered Successfully')
    return res

# 6
def formWord(s, t):
    cnts = Counter(s)
    cntt = Counter(t)
    res = float('inf')
    for k, v in cntt.items():
        res = min(res, cnts[k] // v)
    return res

# print(formWord('mononom', 'mon'))

# 7
def greyness(grid):
    rowsum = [sum(map(int, list(row))) for row in grid]
    colsum = [0] * len(grid[0])
    n, m = len(grid), len(grid[0])
    for j in range(m):
        for i in range(n):
            colsum[j] += int(grid[i][j])
    res = 0
    for i in range(n):
        for j in range(m):
            res = max(res, rowsum[i] + colsum[j] - (m - rowsum[i] + n - colsum[j]))
    return res
def getMaximunGery(grid):
    prfSumRow = [0]*len(grid)
    prfSumCol = [0]*len(grid[0])
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c]=="1":
                prfSumRow[r]+=1; prfSumCol[c]+=1
            else:
                prfSumRow[r]-=1; prfSumCol[c]-=1
    return max(prfSumRow)+max(prfSumCol)

# t = ['1101', '0101', '1010']
# print(greyness(t))
# print(getMaximunGery(t))

# 9


# 27
# 第一题：给你string，要把它分为多少个substring 并且每个substring的element都只出现一次，比如：string="aabcdea"， 可以分为三部分：
def split(s):
    seen = set()
    res = 0
    for c in s:
        if c in seen:
            seen.clear()
            res += 1
        seen.add(c)
    return res + 1
# print(split('aabcdea'))
# print(split('alabama'))
# print(split('aaaaaa'))



# lc 424
def characterReplacement(s: str, k: int) -> int:
    left = 0
    n = len(s)
    cnt = [0] * 26

    def invalid(cnt):
        return sum(cnt) - max(cnt) > k

    res = 0
    for i in range(n):
        c = s[i]
        cnt[ord(c) - ord('A')] += 1
        while left < i and invalid(cnt):
            cnt[ord(s[left]) - ord('A')] -= 1
            left += 1
        res = max(res, i - left + 1)
    return res

# linkedlist non-increasing sublist
class Node:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def fromList(self, list):
        head = Node()
        node = head
        for elem in list:
            node.next = Node(elem)
            node = node.next
        return head.next


def longestNonincreasingSubarray(head):
    prevmaxhead = None
    prevmaxlen = 0
    curmaxhead = head
    curmaxlen = 1
    prev = head
    curr = head.next
    while curr:
        if curr.val > prev.val:
            if curmaxlen > prevmaxlen:
                prevmaxlen = curmaxlen
                prevmaxhead = curmaxhead
            curmaxlen = 1
            curmaxhead = curr
        else:
            curmaxlen += 1
        prev = curr
        curr = curr.next
    res = prevmaxhead if prevmaxlen > curmaxlen else curmaxhead
    reslen = max(prevmaxlen, curmaxlen)
    node = res
    for _ in range(reslen - 1):
        node = node.next
    node.next = None
    return res, reslen

def longestNonincreasingSubarray2(head):
    left = head
    prev = head
    curr = head.next
    reslen = 1
    currlen = 1
    res = head
    while curr:
        if curr.val > prev.val:
            if currlen > reslen:
                reslen = currlen
                res = left
                prev.next = None
            left = curr
        else:
            currlen += 1
        prev = curr
        curr = curr.next

    return res, reslen
# ll = LinkedList().fromList([2,5,4,4,5])
# res, reslen = longestNonincreasingSubarray2(ll)
# print(reslen)
# while res:
#     print(res.val)
#     res = res.next



def buyBook(release):
    heap = []
    min_ = 1
    res = []
    for book in release:
        tmp = []
        heappush(heap, book)
        while heap and heap[0] == min_:
            min_ += 1
            tmp.append(heappop(heap))
        if not tmp: tmp.append(-1)
        res.append(tmp)
    return res
# print(buyBook([1,5,3,2,4]))
# print(buyBook([2,1,4,3]))
# print(buyBook([5,4,3,1,2]))
# print(buyBook([1]))
import random
import functools
def kClosest(points: List[List[int]], k: int) -> List[List[int]]:
    # points.sort(key=lambda x: (x[0] * x[0] + x[1] * x[1], x[0]))
    # return points[:k]
    random.shuffle(points)
    if functools.reduce(lambda x, y: x == y, points):
        return points[:k]
    def compareTo(p1, p2):
        d1 = p1[0] * p1[0] + p1[1] * p1[1]
        d2 = p2[0] * p2[0] + p2[1] * p2[1]
        if d1 == d2:
            return 0
        if d1 > d2:
            return 1
        else:
            return -1

    def partition(arr, left, right):
        wall = left
        pivot = arr[right]
        for i in range(left, right):
            if compareTo(arr[i], pivot) < 0:
                arr[wall], arr[i] = arr[i], arr[wall]
                wall += 1
        arr[wall], arr[right] = arr[right], arr[wall]
        return wall

    left, right = 0, len(points) - 1
    while left < right:
        idx = partition(points, left, right)

        if idx > k - 1:
            right = idx - 1

        else:
            left = idx + 1
        print(idx)
    return points[:k]
# print(kClosest(t, k))

# def klargestComb(nums, k):
#     pos = 0
#     heap = []
#     for num in nums:
#         if num > 0:
#             pos += num
#             heappush(heap, num)
#         else:
#             heappush(heap, -num)
#     res = []
#     while heap:
#         res.append(pos)
#         pos -= heappop(heap)
#     return res
#
# print(klargestComb([3, 5, -2], 6))


def klargestComb(nums, k):
    heap = []
    n = len(nums)
    def bt(start, sum_):
        heappush(heap, sum_)
        if len(heap) > k:
            heappop(heap)
        for i in range(start, n):
            bt(i + 1, nums[i] + sum_)
    bt(0, 0)
    res = []
    while heap:
        res.append(heappop(heap))
    return res[::-1]

# NlgN + klgk
# hard https://leetcode.com/problems/find-the-k-sum-of-an-array/discuss/2456716/Python3-HeapPriority-Queue-O(NlogN-%2B-klogk)
def kSum(nums, k):
    maxSum, absNum = sum(max(0, num) for num in nums), sorted(map(abs, nums))
    heap = [(-maxSum + absNum[0], 0)]
    res = [maxSum]
    for _ in range(k-1):
        nextSum, i = heappop(heap)
        res.append(-nextSum)
        if i + 1 < len(nums):
            heappush(heap, (nextSum - absNum[i] + absNum[i+1], i+1))
            heappush(heap, (nextSum + absNum[i+1], i+1))
    return res



# print(klargestComb([3, 5, -2], 6))
# print(klargestComb([1,2,3,1000], 5))
#
# print(kSum([3, 5, -2], 6))
# print(kSum([1,2,3,1000], 5))
import math

def cohesiveProcess(nums):
    cnt = Counter(nums)
    res = 0
    for k, v in cnt.items():
        res += k
        cohesive = k
        for _ in range(v - 1):
            cohesive = math.ceil(cohesive / 2)
            res += cohesive
    return res


# print(cohesiveProcess([5,5,3,6,5,3]))
# print(cohesiveProcess([4,3,3,3,]))
# print(cohesiveProcess([5,8,4,4,8,2]))


def countPrefectSquareSubset(nums):
    seen = set(nums)
    res = 0
    while seen:
        num = seen.pop()
        tmp = 1
        down = math.sqrt(num)
        up = num ** 2
        while down in seen:
            tmp += 1
            seen.remove(down)
            down = math.sqrt(down)
        while up in seen:
            tmp += 1
            seen.remove(up)
            up = up ** 2
        res = max(res, tmp)
    return res
# print(countPrefectSquareSubset([2, 8, 9, 16, 4, 3]))


def commonPrefix(s):
    n = len(s)
    res = 0
    for i in range(n):
        current = 0
        for j in range(i, n):
            if s[j] == s[j - i]:
                current += 1
            else:
                break
        res += current
    return res

# print(commonPrefix('abcabcd'))

# TODO O(n)
def student_imbal(nums):
    n = len(nums)
    res = 0
    for i in range(n):
        min_, max_ = nums[i], nums[i]
        for j in range(i+1, n):
            min_ = min(min_, nums[j])
            max_ = max(max_, nums[j])
            res += j - i < max_ - min_
    return res

# print(student_imbal([4,1,3,2]))
# print(student_imbal([4,3,1,2]))

# 9
def groupMoive(awards, k):
    awards.sort()
    res = 1
    curmin = awards[0]
    for i in range(1, len(awards)):
        if awards[i] - curmin > k:
            curmin = awards[i]
            print(curmin)
            res += 1
    return res

# print(groupMoive([1,5,4,6,8,2], 3))


# 10
# 970, 2104, shipment imbalance (max - min of all subarray)
def subArrayRanges(nums: List[int]) -> int:
    def sumSubarrayMins(arr: List[int]) -> int:
        stack = [0]
        arr = [0] + arr
        n = len(arr)
        result = [0] * n
        for i in range(n):
            while stack and arr[stack[-1]] > arr[i]:
                stack.pop()
            j = stack[-1] if stack else 0
            result[i] = result[j] + arr[i] * (i - j)
            stack.append(i)
        return sum(result)

    def sumSubarrayMaxs(arr: List[int]) -> int:
        stack = [0]
        arr = [0] + arr
        n = len(arr)
        result = [0] * n
        for i in range(n):
            while stack and arr[stack[-1]] < arr[i]:
                stack.pop()
            j = stack[-1] if stack else 0
            result[i] = result[j] + arr[i] * (i - j)
            stack.append(i)
        return sum(result)

    return sumSubarrayMaxs(nums) - sumSubarrayMins(nums)

def minStockNet(stockPrice):
    total = sum(stockPrice)
    res = float('inf')
    n = len(stockPrice)
    ans = -1
    presum = 0
    for i in range(n-1):
        presum += stockPrice[i]
        left = presum // (i + 1)
        right = (total - presum) // (n-i-1)
        if abs(right - left) < res:
            res = abs(right - left)
            ans = i + 1
    return ans

# print(minStockNet([1,3,2,3]))
# print(minStockNet([1,1,1,1,1,1]))

# 13
def maxSubarraySum(nums, k):
    sum_ = 0
    res = 0
    n = len(nums)
    for i in range(k):
        sum_ += nums[i]
    for j in range(k, n):
        res = max(res, sum_)
        sum_ -= nums[j-k]
        sum_ += nums[j]
    res = max(sum_, res)
    return sum(nums) - res

# print(maxSubarraySum([7,3,6,1], 2))

# 14, lc1846
def maximumElementAfterDecrementingAndRearranging(arr: List[int]) -> int:
    arr.sort()
    prev = 0
    for i in range(len(arr)):
        prev = min(prev + 1, arr[i])
    return prev

# 15

# 17
# node=ListNode(0,None) （会报错！）要分开来：node=ListNode(0)   node.next=None
def llprint(head):
    res = []
    while head:
        res.append(head.val)
        head = head.next
    print('->'.join(res))

def geek(head, operations):
    dummy = Node(-1)
    dummy.next = head
    tail = dummy
    while tail.next:
        tail = tail.next
    for op in operations:
        action, data = op[0], op[1]
        if action == 'PUSH_TAIL':
            tail.next = Node(data)
            tail = tail.next
            llprint(dummy.next)
        elif action == 'PUSH_HEAD':
            dummy_next = dummy.next
            dummy.next = Node(data)
            dummy.next.next = dummy_next
            llprint(dummy.next)
        elif action == 'POP_HEAD':
            dummy.next = dummy.next.next
            llprint(dummy.next)
    return dummy.next
#
# ll = LinkedList().fromList(['pen', 'cup'])
# geek(None, [['PUSH_TAIL', 'fan'], ['PUSH_HEAD', 'jam'], ['POP_HEAD', ''], ['POP_HEAD', '']])


# 21
def maxCluster(booting, processing, k):
    sum_ = left = 0
    maxheap = []
    res = 0
    n = len(booting)
    def invalid():
        return -maxheap[0][0] + sum_ * (i - left + 1) > k
    for i in range(n):
        sum_ += processing[i]
        heappush(maxheap, [-booting[i], i])
        while invalid():
            while maxheap[0][1] <= left:
                heappop(maxheap)
            sum_ -= processing[left]
            left += 1
        if i - left + 1 >= res:
            print(i, left, -maxheap[0][0] + sum_ * (i - left + 1))
            print(maxheap)
        res = max(res, i - left + 1)
    return res

def maxClusterDeque(booting, processing, k):
    sum_ = res = 0
    q = deque([])
    left = 0
    n = len(booting)
    def invalid():
        return booting[q[0]] + sum_ * (i - left + 1) > k

    for i in range(n):
        sum_ += processing[i]
        while q and booting[q[-1]] <= booting[i]:
            q.pop()
        q.append(i)
        while invalid():
            if q[0] <= left:
                q.popleft()
            sum_ -= processing[left]
            left += 1
        if i - left + 1 >= res:
            print(i, left, booting[q[0]] + sum_ * (i - left + 1))
            print(q)
        res = max(res, i - left + 1)
    return res

# print(maxClusterDeque([3,6,1,3,4], [2,1,3,4,5], 25))
# print('---')
# print(maxClusterDeque([3,6,1,3,4][::-1], [2,1,3,4,5][::-1], 46))


def consecutiveDecreasing(nums):
    nums = [0] + nums
    res = []
    n = len(nums)
    i = 0
    while i < n:
        start = i
        while i < n and (nums[i-1] == nums[i] + 1):
            i += 1
        if start == i:
            i += 1
        else:
            res.append(i - start + 1)
    return res

# print(consecutiveDecreasing([4,3,2,5,9,8]))
# print('*****')
# print(consecutiveDecreasing([9,3,2,5,9,8]))


# 24 TODO


# 26
def missingNumber(nums=[1,3,5,7,10], k=7):
    n = len(nums)
    s = set(nums)
    res = []
    for i in range(1, k+1):
        if i not in s:
            res.append(i)
        if len(res) == k - n:
            break

    return sum(res)
# print(missingNumber())


# 35
def tempChange(nums):
    prefix = [0]
    sum_ = 0
    for num in nums:
        sum_ += num
        prefix.append(sum_)
    res = 0
    for i in range(len(nums)):
        res = max(res, max(prefix[i+1], prefix[-1] - prefix[i]))
    return res
# print(tempChange([6, -2, 5]))


def memorySegment(nums=[10,4,8,13,20], k=2):
    sum_ = 0
    for i in range(k):
        sum_ += nums[i]
    res = sum_
    for i in range(k, len(nums)):
        sum_ += nums[i] - nums[i-k]
        res = max(res, sum_)
    return sum(nums) - res
# print(memorySegment())

def pMatching(s='acaccaa', t='aac', p=2):
    ns = len(s)
    nt = len(t)
    def check(ss):
        print(ss)
        if len(ss) < nt:return
        cntt = Counter(t)
        cnts = Counter()
        for i in range(nt):
            cnts[ss[i]] += 1
        res = 1 if cntt == cnts else 0
        for i in range(nt, len(ss)):
            cnts[ss[i]] += 1
            cnts[ss[i-p]] -= 1
            res += 1 if cntt == cnts else 0
        print(res)
        return res
    return sum(check(s[i:ns:p]) for i in range(p))

# print(pMatching())


# 43 47 Z Sequence  TODO

# student imbal
def find_imbalances(rank):
    count = 0
    s = set()
    for i in range(len(rank)):
        s.clear()
        max = rank[i]
        min = rank[i]
        s.add(rank[i])
        for j in range(i+1, len(rank)):
            if rank[j] - 1 in s and rank[j] + 1 in s:
                pass
            elif min > rank[j] and rank[j] + 1 in s:
                min = rank[j]
            elif max < rank[j] and rank[j] - 1 in s:
                max = rank[j]
            elif min > rank[j]:
                min = rank[j]
                count += 1
            elif max < rank[j]:
                max = rank[j]
                count += 1
            elif rank[j] - 1 in s or rank[j] + 1 in s:
                count += 1
            else:
                count += 2
            s.add(rank[j])
            print(count, s)
    return count

# print(find_imbalances([4,1,5,3,2]))
def student_bf(nums):
    n = len(nums)
    res = 0
    for i in range(n-1):
        for j in range(i+1, n+1):
            tmp = sorted(nums[i:j])
            print(nums[i:j])
            if len(tmp) >= 2 and tmp[-1] != tmp[0] + len(tmp) - 1:
                print('^')
                res += 1
    return res
print(student_bf([4,1,5,3,2]))
def numberSubarrayMinMaxk(nums, k):
    maxheap, minheap = [], []
    n = len(nums)
    left = -1
    res= 0
    for i, num in enumerate(nums):
        heappush(maxheap, (-num, i))
        heappush(minheap, (num, i))
        while maxheap and minheap and -maxheap[0][0] - minheap[0][0] > k:
            if maxheap[0][0] == -num:
                if minheap:
                    _, left = heappop(minheap)
            else:
                if maxheap:
                    _, left = heappop(maxheap)
        # print(minheap, maxheap)
        if maxheap and minheap:
            res += i - left
    return res

# print(numberSubarrayMinMaxk([1,3,6,8], 3))



def studentImbal(nums):
    n = len(nums)
    stack = []
    index2val = {num: i for i, num in enumerate(nums)}
    left = [0] * n
    right = [0] * n
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            stack.pop()
        left[i] = stack[-1] if stack else -1
        stack.append(i)
    stack.clear()
    for i in range(n)[::-1]:
        while stack and nums[stack[-1]] < nums[i]:
            stack.pop()
        right[i] = stack[-1] if stack else n
        stack.append(i)
    cnt = 0
    print(left, right)
    for i in range(n):
        if nums[i] < n - 1:
            addoneidx = index2val[nums[i] + 1]
            if addoneidx > i:
                cnt += (i + 1) * (addoneidx - right[i])
                if left[i] >= 0:
                    cnt += (left[i] + 1) * (right[i] - i)
            else:
                cnt += (left[i] - addoneidx) * (n-i)
                if right[i] < n:
                    cnt += (i - left[i]) * (n - right[i])
    return cnt


print(studentImbal([4,1, 5,3,2]))
def maxCount(s):
    c2maxcnt = defaultdict(list)
    m = Counter()

    for i, c in enumerate(s):
        m[c] += 1
        maxval = max(m.values())
        for k in m.keys():
            if maxval == m[k]:
                c2maxcnt[k].append(i)
    print(c2maxcnt)
    return max(map(len, c2maxcnt.values()))

# https://leetcode.com/problems/minimum-rounds-to-complete-all-tasks/discuss/1955622/JavaC%2B%2BPython-Sum-up-(freq-%2B-2)-3
# print(maxCount('bccaaacb'))
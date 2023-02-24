from functools import lru_cache as cache

from utils import *


def reach(nums):
    num2idx = defaultdict(list)
    for i, num in enumerate(nums):
        num2idx[num].append(i)
    # print(num2idx)
    n = len(nums)
    q = deque([0])
    visited = set()
    visited.add(0)
    level = 0
    while q:
        sz = len(q)
        for _ in range(sz):
            node = q.popleft()
            if node == n - 1:
                return level
            for next_idx in num2idx[nums[node]] + [node - 1, node + 1]:
                if 0 <= next_idx < n and next_idx not in visited:
                    q.append(next_idx)
                    visited.add(next_idx)
        level += 1
    return -1

# print(reach([1, 2, 3, 10, 11, 12, 1, 2, 3]))
# print(reach([1, 8, 5, 7, 3, 8, 12, 15, 0, 3]))
# print(reach([1, 8, 4, 6, 11, 32, 3, 8, 5, 7, 3, 7, 8, 12, 15, 0, 3]))


# todo leaderboard
class Leaderboard:
    def __init__(self):
        self.time2record = SortedDict()
        self.id2record = defaultdict(lambda : [0,0,0])
    def submission(self, sid, iscorrect, time):
        ncorrect, ntotal, time_total = self.id2record[sid]
        self.id2record[sid][0] += iscorrect
        self.id2record[sid][1] += 1
        self.id2record[sid][2] += time

        if ntotal > 0:
            avg_old = time_total / ntotal
            self.leaderboard_remove(sid, avg_old)
        avg_new = (time_total + time) / (ntotal + 1)
        self.leaderboard_add(sid, avg_new)

    def leaderboard_remove(self, sid, avg_old):
        if avg_old in self.time2record and sid in self.time2record[avg_old]:
            self.time2record[avg_old].remove(sid)

    def leaderboard_add(self, sid, avg_new):
        if avg_new not in self.time2record:
            self.time2record[avg_new] = set()
        self.time2record[avg_new].add(sid)

    def display(self):
        res = []
        for avg_time in self.time2record:
            for sid in self.time2record[avg_time]:

                ncorrect, ntotal, time_total = self.id2record[sid]
                if ncorrect / ntotal >= .75:
                    res.append((sid, ncorrect / ntotal, time_total / ntotal))
        return res


# lb = Leaderboard()
# lb.submission(0, 1, 10)
# lb.submission(0, 0, 20)
# lb.submission(0, 1, 10)
# lb.submission(0, 1, 30)
# print(lb.display())
# lb.submission(2, 0, 2)
# lb.submission(1, 1, 20)
# print(lb.display())
# lb.submission(1, 1, 1)
# print(lb.display())
# lb.submission(1, 0, 1)
# print(lb.display())
#
#

# c=[5,6,7,8,8,10]
# t=[1,1,1,1,1,10]
# n = len(c)
# @cache
# def sufsum(i):
#      return t[i]+sufsum(i+1) if i<len(t) else 0
#
# @cache
# def f(i,j):
#     if j+sufsum(i) < 0:
#         return float('inf')
#
#     if j>=n-i:
#         return 0
#
#     return min(c[i]+f(i+1,j+t[i]), f(i+1,j-1))
# for i in range(n+5):
#     print(sufsum(i))
# print(f(0,0))


def getMaxBarrier(initialEnergy, th):
      maxEnergy = max(initialEnergy)
      left = 0
      right = maxEnergy
      def getSum(barrier):
          ans = 0
          for i in initialEnergy:
                    if i-barrier > 0:
                          ans += i-barrier
          return ans
      while left <= right:
          mid = (right-left)//2 +left
          sum_m = getSum(mid)
          if sum_m == th:
                    return mid
          elif sum_m > th:
                    left = mid+1
          else:
                    right = mid-1
      return right
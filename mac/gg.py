from tqdm import  tqdm
from typing import *
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
print(findMaximums([0,1,2,4]))
print(findMaximums([1,2,5,1]))

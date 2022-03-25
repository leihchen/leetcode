import random
from bisect import bisect_right, bisect_left, insort
print(random.random())
def getMostVisited(n, sprints):
    m = len(sprints)
    planes = []
    for i in range(m-1):
        start, end = min(sprints[i], sprints[i+1]), max(sprints[i], sprints[i+1])
        planes.append((start, 1))
        planes.append((end, -1))
    planes.sort(key=lambda x: (x[0], -x[1]))
    max_, res = 0, 0
    cur = 0
    for ts, op in planes:
        cur += op
        if cur > max_:
            max_ = cur
            res = ts
    return res

# print(getMostVisited(5, [2, 4, 1, 3]))
# print(getMostVisited(10, [1, 5, 10, 3]))
def oddEven(nums):
    wall = 0
    n = len(nums)
    for i in range(n):
        if nums[i] % 2 == 1:
            nums[i], nums[wall] = nums[wall], nums[i]
            wall += 1
t1 = [1,345,342,6,243562,7,1214]
t2 = [3245,5,2,4,62,6,2,7,27,821,4,7]
# oddEven(t1); print(t1)
# oddEven(t2); print(t2)

def matrixSummation(after):
    n, m = len(after), len(after[0])
    before = [[0] * m for _ in range(n)]
    before[0][0] = after[0][0]
    for i in range(n):
        for j in range(m):
            before[i][j] = after[i][j] + (0 if i == 0 or j == 0 else \
                after[i-1][j-1]) - (0 if i == 0 else after[i-1][j]) - (0 if j == 0 else after[i][j-1])
    return before
t = [[2, 5], [7, 17]]
print(matrixSummation(t))

def threeSumSmaller(nums, tar):
    nums.sort()
    n = len(nums)
    res = 0
    for i in range(n-2):
        left = i + 1
        right = n - 1
        while left < right:
            if nums[i] + nums[left] + nums[right] < tar:
                res += right - left
                left += 1
            else:
                right -= 1
    return res
print(threeSumSmaller([5, 1, 3, 4, 7], 12))

def my_bisect_right(nums, tar):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] <= tar:
            left = mid + 1
        else:
            right = mid - 1
    return left
l, t = [1,1,2,3,5], 1
print(my_bisect_right(l, t), bisect_right(l, t))
assert my_bisect_right(l, t) == bisect_right(l, t)
print(bisect_left([0, 1, 1, 3], 1))
print(bisect_right([0, 1, 1, 3], 1))

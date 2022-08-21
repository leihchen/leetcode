from typing import *
from collections import *
import math
from bisect import bisect_left
import random
from random import randint
from copy import deepcopy
from heapq import heappush, heappop, heapify, nlargest
from itertools import combinations


# 1155
# 翻转链表2, coin change 2

# 3
# 151
# 用static array设计 dynamic array



# sd + ood类似设计一个权限管理系统，有user role和permission，有一个映射关系，设计assign, revoke和get permission的接口，然后要mock db te‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌sting类似
# 350

# 528 followup要求get方法O1
# 355 followup如果post重复tweet怎么处理
# 17 followup如果给一个字典，只能输出字典里有的单词
# dp下楼梯
# quickselect
#

# You're given an int array that each number appears twice except the a specific number, which you'll need to return its index. Note that the same number always appear in pairs, for example: [2334455] the output should be 0, [3344566] the output should be 4.
def findSingle(nums):
    n = len(nums)
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if mid in (n-1, 0):
            return mid
        is_end = nums[mid+1] != nums[mid]
        if is_end and mid % 2 == 1:
            left = mid + 1
        elif is_end and mid % 2 == 0:
            right = mid - 1
        elif not is_end and mid % 2 == 0:
            left = mid + 1
        else:
            right = mid - 1
    return left

#                  0 1 2 3 4 5 6
print (findSingle([3,3,4,4,7,5,5]))


# lc 给一个String[][], 包括起飞城市，落地城市，及票价，要求返回城市A -> B的最低票价，并且返回path。并且有一个限制是最多乘坐k次飞机。
# 143. Reorder List


# Path sum III
# 833. Find And Replace in String
# 1155. Number of Dice Rolls With Target Sum

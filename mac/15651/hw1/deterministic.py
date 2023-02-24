import math

n, k = tuple([int(x) for x in input().split()])
vals = [int(x) for x in input().split()]

BLOCK_SIZE = 5


def get_median(nums):
    n = len(nums)
    return sorted(nums)[n // 2]  # pick the larger one


def median_of_median(nums):
    n = len(nums)
    # base case
    if n <= 5: return get_median(nums)
    # recursive case
    nblock = math.ceil(n / BLOCK_SIZE)
    blocks = [get_median(nums[i * BLOCK_SIZE: (i + 1) * BLOCK_SIZE]) for i in range(nblock)]
    return median_of_median(blocks)


def deterministic_select(nums, k):
    pivot = median_of_median(nums)
    less, greater = [], []
    for num in nums:
        if num < pivot:
            less.append(num)
        elif num > pivot:
            greater.append(num)
    if len(less) == k - 1 or (len(less) < k - 1 and len(nums) - len(greater) >= k):
        return pivot
    elif len(less) > k - 1:
        return deterministic_select(less, k)
    else:
        return deterministic_select(greater, k - (len(nums) - len(greater)))

# vals, k = [3,2,3,1,2,4,5,5,6]
# vals, k = [53, 12], 1
# vals, k = [6, 8, 12, 4, 2], 3
# vals, k = [3,3,3,3,4,3,3,3,3], 9
print(deterministic_select(vals, k))



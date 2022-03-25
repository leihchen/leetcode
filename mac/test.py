from typing import *

def findPeakElement(values: List[int]) -> int:
    # find the first decreasing point
    n = len(values)
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2  # (left + right) // 2
        if values[left] < values[mid] < values[right]:
            left = mid + 1
        elif values[left] > values[mid] > values[right]:
            right = mid - 1
        elif values[left] < values[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return values[left]
print(findPeakElement([1,2,1,3,5,6,4]))

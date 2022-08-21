from typing import *
from collections import *
import math
from bisect import bisect_left
import random
from heapq import heappush, heappop, heapify
from itertools import combinations



# 427. Construct Quad Tree
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight

def construct(grid: List[List[int]]) -> 'Node':
    def allSame(grid):
        for row in grid:
            for elem in row:
                if grid[0][0] != elem:
                    return False
        return True

    if allSame(grid):
        return Node(grid[0][0], True, None, None, None, None)
    n = len(grid)
    topLeft = [row[:n // 2] for row in grid[: n // 2]]
    topRight = [row[n // 2:] for row in grid[: n // 2]]
    bottomLeft = [row[:n // 2] for row in grid[n // 2:]]
    bottomRight = [row[n // 2:] for row in grid[n // 2:]]
    return Node(grid[0][0], False, construct(topLeft), construct(topRight), construct(bottomLeft),
                construct(bottomRight))



# 465 Optimal Account Balancing
def minTransfers(transactions):
    """
    Get all persons that have outstanding balance + or -, and discard others
    Greedily find smallest group of persons whose balances sum to zero, and set them to zero.
    Do this by getting all combinations of size 2, then size 3 etc. The moment any size group sums to zero,
    recurse on remaining outstanding persons. Because a greedy solution is guaranteed to exist and is optimal
    e.g.
    6->0 : 50, 1->6 : 40, 2->6 : 10, 6->3 : 40, 6->4 : 40, 5->6 : 25, 6->7 : 10, 7->1 : 10
    balance : 0:50, 1:-30, 2:-10, 3:40, 4:10, 5:-25, 6:-35, 7:0
    remove 7 since it's already at 0, and participating in any give/take only increases transaction size
    Then try to find any combination of size 2 that can cancel each other's debts
    e.g. 2 & 4. 4 gives 10 to 2, 1 transaction. Then 2 & 4's balances are set to 0, then remove them.
    Recurse on remaining.
    Again try to find any combinations of size 2. None. So find combinations of size 3, size 4, size 5.
    size 5 - all remaining add up to 90 - 90. 4 transactions
    Total 5 transactions
    """

    def outstanding(balance):  # Get all persons with either + or - money
        return {person: money for person, money in balance.items() if money != 0}

    def min_trans(balance):
        if not balance:
            return 0

        for size in range(2, len(balance) + 1):  # Greedily start from smallest size of 2
            for group in combinations(balance.keys(), size):  # Get all combinations
                if sum(balance[person] for person in group) == 0:  # If they can cancel each others debts
                    for person in group:
                        balance[person] = 0  # Set all persons in that group to 0
                    transactions = size - 1  # Number of transactions needed is size-1
                    return transactions + min_trans(outstanding(balance))  # Recurse on remaining outstanding balances

    balance = defaultdict(int)
    for u, v, money in transactions:  # Get final balance of each person
        balance[u] -= money
        balance[v] += money

    return min_trans(outstanding(balance))

#


# 399. Evaluate Division

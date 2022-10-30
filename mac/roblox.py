# https://leetcode.com/problems/random-pick-with-weight/submissions/
from utils import *
class Solution:
    def __init__(self, w: List[int]):
        # [1,3]
        # [0,1,1,1,] 1/4
        # -> [1,4] -> [1,4] ->
        # 0 x <= 1, 1 x <= 4 < 1
        # first larger index than x
        self.presum = []
        sum_ = 0
        for element in w:
            sum_ += element
            self.presum.append(sum_)

    def pickIndex(self) -> int:
        x = random.randint(1, self.presum[-1])
        return bisect_left(self.presum, x)

# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()

# https://leetcode.com/problems/minimum-number-of-frogs-croaking/
class Solution:
    def minNumberOfFrogs(self, croakOfFrogs: str) -> int:
        cnt = 0
        n = len(croakOfFrogs)
        mapper = {c:i for i, c in enumerate('croak')}
        s = [mapper[c] for c in croakOfFrogs]
        progress = Counter()  # next char to # of thread
        sz = 0
        res = 0
        for c in s:
            if c == 0:
                progress[1] += 1
                sz += 1
                res = max(res, sz)
            else:
                if progress[c] > 0:
                    progress[c] -= 1
                    if c + 1 != 5:
                        progress[c+1] += 1
                    else:
                        sz -= 1
                else:
                    return -1
        return res if sz == 0 else -1

# https://leetcode.com/problems/spiral-matrix-ii/
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        num = 1
        left, right, top, bottom = 0, n - 1, 0, n - 1
        res = [[0] * n for _ in range(n)]

        while right - left >= 0 or bottom - top >= 0:
            for i in range(left, right + 1):
                res[top][i] = num
                num += 1
            top += 1
            for i in range(top, bottom + 1):
                res[i][right] = num
                num += 1
            right -= 1
            for i in range(right, left - 1, -1):
                res[bottom][i] = num
                num += 1
            bottom -= 1
            for i in range(bottom, top - 1, -1):
                res[i][left] = num
                num += 1
            left += 1
        return res


# TODO
# 41. First Missing Positive
# https://leetcode.com/problems/first-missing-positive/
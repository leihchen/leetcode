class Solution:
    i = 0
    def expand(self, s: str) -> str:
        if self.i >= len(s):
            return ''
        res = []
        while self.i < len(s):
            if ord('a') <= ord(s[self.i]) <= ord('z'):
                res.append(s[self.i])
                self.i += 1
            elif s[self.i] == ')':
                break
            elif s[self.i] == '(':
                self.i += 1
                tmp = self.expand(s)
                self.i += 4
                res.append(tmp * int(s[self.i - 2]))
        return ''.join(res)

soln = Solution()
print(soln.expand('(ab(d){3}){2}'))
soln2 = Solution()
print(soln2.expand('(ab){3}(cd){2}'))

def queen_attack(x1, y1, x2, y2):
    return x1 == x2 or y1 == y2 or x1 + y1 == x2 + y2 or x1 - y1 == x2 - y2

def count_1s(n):
    res = 0
    while n:
        n = n & (n-1)
        res += 1
    return res

def maxSubarray(s):
    res = float('-inf')
    tmp = float('-inf')
    for i in s:
        tmp = max(tmp + i, i)
        res = max(res, tmp)
    return res

class Solution:
    def maximumDifference(self, nums) -> int:
        res = float('-inf')
        min_ = nums[0]
        for i in range(1, len(nums)):
            res = max(res, nums[i] - min_)
            min_ = min(min_, nums[i])
        return res if res > 0 else -1


class Solution:
    def validIPAddress(self, IP: str) -> str:
        if IP.count('.') == 3:
            for i in IP.split('.'):
                try:
                    sub = int(i)
                except:
                    return "Neither"
                if sub < 0 or sub > 255 or str(sub) != i:
                    return "Neither"
            return "IPv4"
        elif IP.count(':') == 7:
            for i in IP.split(':'):
                try:
                    sub = int(i, 16)
                except:
                    return "Neither"
                if len(i) > 4 or len(i) == 0:
                    return "Neither"
            return "IPv6"
        else:
            return "Neither"


class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or len(s) == 0 or s[0] == '0': return 0
        dp = [0] * len(s)
        dp[0] = 1
        for i in range(1, len(s)):
            if 10 <= int(s[i-1:i+1]) <= 26:
                dp[i] += dp[i-2] if i - 2 >= 0 else 1
            if s[i] != '0':
                dp[i] += dp[i-1]
        # print(dp)
        return dp[-1]


def chocolates(nums):
    n = len(nums)
    dp = [0] * n
    for i in range(n):
        dp[i] = nums[i]
        add = 0
        for j in range(i-1):
            add = max(add, dp[j])
        dp[i] += add
    print(dp)
    return max(dp)

def chocolates(choco):
    n = len(choco)
    max_ = 0
    dp = [0] * n
    for i in range(n):
        dp[i] = max(choco[i], choco[i] + max_)
        if i >= 1:
            max_ = max(max_, dp[i - 1])
    return max(dp)
print(chocolates([5,30,99,60,5,10]))
# counter-clockwise 90: zip(*matrix)[::-1]
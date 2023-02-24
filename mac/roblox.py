# https://leetcode.com/problems/random-pick-with-weight/submissions/
import random

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

# https://leetcode.com/problems/implement-trie-prefix-tree/
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isWord = False


class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for c in word:
            node = node.children[c]
        node.isWord = True

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            if c not in node.children:
                return False
            node = node.children[c]
        return node.isWord

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            if c not in node.children:
                return False
            node = node.children[c]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

# https://leetcode.com/problems/rotate-image/
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        matrix[::] = zip(*matrix[::-1])


def candyCrushInit(height, width, colors):
    if height == 0:
        return []
    grid = candyCrushInit(height-1, width, colors)
    curr_row = []
    candidate = set(colors)
    for i in range(width):
        ban = set()
        if i >= 2 and curr_row[i-2] == curr_row[i-1]:
            ban.add(curr_row[i-1])
        if height-1 >= 2 and grid[-1][i] == grid[-2][i]:
            ban.add(grid[-1][i])
        this_candidate = tuple(candidate - ban)
        curr_row.append(this_candidate[random.randint(0, len(this_candidate)-1)])
    grid.append(curr_row)
    return grid

# for row in candyCrushInit(4,4,[0,1,2]):
#     print(row)

def majhong(cards):
    c = Counter(cards)
    res = []
    for i in sorted(c):
        if c[i] > 0:
            straight = True
            need = c[i]
            for j in range(3):
                if c[i + j] < need:
                    straight = False
            if straight:
                res.append( (str(i) + str(i+1) + str(i+2)) * need)
                for j in range(3):
                    c[i + j] -= need
    pairs = 0
    for i in c:
        if c[i] % 3 == 0:
            res.append(str(i) * c[i])
            c[i] = 0
        elif c[i] % 2 == 0:
            # res.append(str(i) * (i // 3) * 3)
            res.append(str(i) * 2)
            c[i] -= 2
            pairs += 1

    print(c)
    print(res)

    return pairs >= 1

# print(majhong([1,1,1,2,3,4,6,6,7,7,8,8,9,9]))
# print(majhong([1,1, 1,2,3, 2,3,4, 3,3,3, 4,4,4]))
# TODO
# 41. First Missing Positive
# https://leetcode.com/problems/first-missing-positive/




# 第一轮 tech：1419
# 第二轮 tech：trie

# lc528 + lc41的简单版

# 第一轮（coding 45min）：m*n grid，1表示可以走的地方，0表示obstacle，
# 人物从左上角出发，上下左右四个方向，可以走相邻的1或者跳过一个0到1，
# 判断人物能不能从左上角走到右下角
# 如果人物可以“助跑”，连续在同一个方向走过k个1就可以跳过k个0，怎么修改

# 第二轮 （coding 45min）：2个list，一个表示keyword-》content type，
# 另一个表示content type -》instructions，给一个list of keywords，
# 生成一个email text containing instructions 比较trivial，
# 主要考string processing 有意思的一点是在过程中混着问了一些bq，
# 比如说当遇到product manager给的instruction list没有想要的content
# type的时候应该怎么做 强烈建议用python写 用cpp有点折磨而且容易出错


# for a cell, larger continous run of 1 is better than shorter
# we can visit a cell twice only if the new visit is a better visit
# direction has no preference
# get next O(n),
# cell_x, cell_y, steak, dir
# visited[0] = [0,0,0,0] streak in four direction,
# if visited[x][y][dir] < streak, then visit
# this ensures termination of the algo
# ie. going back (loop) won't happend since streak is not increased
def streakMaze(grid):
    n, m = len(grid), len(grid[0])
    visited = [[[-1,-1,-1,-1] for _ in range(m)] for _ in range(n)]
    dirs = [[0,1], [1,0], [0,-1], [-1,0]]
    q = deque([[0,0,0,-1]])
    def getNext(x, y, streak, dir):
        if dir in (0, 2):
            dy = dirs[dir][1]
            for ny in range(1, streak+1):
                newy = y + ny * dy + 1
                newx = x
                if 0 <= newx < n and 0 <= newy < m and grid[newx][newy] == 1:
                    yield newx, newy, 0, -1
        else:
            dx = dirs[dir][0]
            for nx in range(1, streak+1):
                newx = x + nx * dx + 1
                newy = y
                if 0 <= newx < n and 0 <= newy < m and grid[newx][newy] == 1:
                    yield newx, newy, 0, -1

        for next_dir, (dx, dy) in enumerate(dirs):
            newx, newy = x + dx, y + dy
            if 0 <= newx < n and 0 <= newy < m:
                if grid[newx][newy] == 1:
                    if next_dir == dir:
                        # increase streak
                        yield newx, newy, streak + 1, next_dir
                    else:
                        yield newx, newy, 1, next_dir

    while q:
        x, y, streak, dir = q.popleft()
        if x == n-1 and y == m-1:
            return True
        for newx, newy, news, newdir in getNext(x, y, streak, dir):
            if visited[newx][newy][newdir] < news:
                visited[newx][newy][newdir] = news
                q.append([newx, newy, news, newdir])
    # print(visited)
    return False
grid = [
    [1,1,1,0,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0],
    [0,0,0,0,0,1]
]
print(streakMaze(grid))

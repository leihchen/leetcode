import math
import sys


class SegTree:
    def LeftChild(self, u: int):
        return 2 * u + 1

    def RightChild(slef, u: int):
        return 2 * u + 2

    def Parent(self, u: int):
        return (u - 1) // 2

    def __init__(self, a, reducer=lambda x, y: x + y):
        self.reducer = reducer

        def build(u: int, l: int, r: int):
            mid = (l + r) // 2
            if self.LeftChild(u) < self.n - 1:
                build(self.LeftChild(u), l, mid)
            if self.RightChild(u) < self.n - 1:
                build(self.RightChild(u), mid, r)
            lres = A[self.LeftChild(u)]
            rres = A[self.RightChild(u)]
            A[u] = self.reducer(lres, rres)

        self.n = len(a)
        A = [(0, 0, 0)] * (2 * self.n - 1)
        for i in range(self.n):
            A[i + self.n - 1] = a[i]
        build(0, 0, self.n)

        self.A = A

    def assign(self, i, x):
        u = i + self.n - 1
        self.A[u] = x
        while u > 0:
            u = self.Parent(u)
            self.A[u] = self.reducer(self.A[self.LeftChild(u)], self.A[self.RightChild(u)])

    def rangeSum(self, i, j):
        def compute_sum(u, i, j, L, R):
            if i <= L and R <= j:
                return self.A[u]
            else:
                mid = (L + R) // 2
                if i >= mid:
                    return compute_sum(self.RightChild(u), i, j, mid, R)
                elif j <= mid:
                    return compute_sum(self.LeftChild(u), i, j, L, mid)
                else:
                    left_sum = compute_sum(self.LeftChild(u), i, j, L, mid)
                    right_sum = compute_sum(self.RightChild(u), i, j, mid, R)
                    return self.reducer(left_sum, right_sum)

        return compute_sum(0, i, j, 0, self.n)

    # def get_range(self, i, j):
    #     return self.A[i + self.n - 1:j + self.n - 1]


MOD1 = 909259097
# MOD2 = 97
# MOD = 1000000007
radix = 62


def barxy(x, y, ny):
    return ((pow(radix, ny, MOD1) * x) % MOD1 + y) % MOD1
def hash(x, y):
    # return hash(x+y)
    modxf, modxr, nx = x
    modyf, modyr, ny = y

    return ((pow(radix, ny, MOD1) * modxf) % MOD1 + modyf) % MOD1, \
           ((pow(radix, nx, MOD1) * modyr) % MOD1 + modxr) % MOD1, \
           nx + ny


APLHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
d = {c: i for i, c in enumerate(APLHABET)}


def c2i(c):
    return d[c]


class DynamicPalindrome:
    def reverse_indicer(self, i):
        return self.n - i - 1

    def __init__(self, input):
        x = len(input)
        next_power_of_2 = pow(2, math.ceil(math.log2(x)))
        padding = next_power_of_2 - x
        self.n = next_power_of_2
        input = [(c2i(c), c2i(c), 1) for c in input] + [(0, 0, 0)] * padding
        self.forward = SegTree(input, reducer=hash)
        # self.reverse = SegTree(input[::-1], reducer=hash)

    def modify(self, i, c):
        self.forward.assign(i, (c2i(c), c2i(c), 1))
        # self.reverse.assign(self.reverse_indicer(i), (c2i(c), c2i(c), 1))

    def query(self, i, j):
        f = self.forward.rangeSum(i, j + 1)
        # r = self.reverse.rangeSum(self.reverse_indicer(j + 1) + 1, self.reverse_indicer(i) + 1)
        return f[0] == f[1]

    def query2(self, i, j):
        f = self.forward.rangeSum(i, j + 1)
        # r = self.reverse.rangeSum(self.reverse_indicer(j + 1) + 1, self.reverse_indicer(i) + 1)
        return f

    def pick(self, i, k, j):
        if i == k == j:
            return True
        if i == k:
            return self.query(k + 1, j)
        elif j == k:
            return self.query(i, k - 1)
        x, y, nleft = self.query2(i, k - 1)
        s, t, nright = self.query2(k + 1, j)
        xs = barxy(x, s, nright)
        ty = barxy(t, y, nleft)
        return xs == ty

    # def get_range(self, i, j):
    #     f = self.forward.get_range(i, j + 1)
    #     r = self.reverse.get_range(self.reverse_indicer(j + 1) + 1, self.reverse_indicer(i) + 1)
    #     print(''.join(APLHABET[x] for x, _ in f))
    #     print(''.join(APLHABET[x] for x, _ in r[::-1]))


def execute_test(file=sys.stdin):
    _, niter = file.readline().strip().split()
    input = file.readline().strip()
    dp = DynamicPalindrome(input)
    log = []
    for _ in range(int(niter)):
        line = file.readline()
        argv = line.strip().split()
        if argv[0] == 'Q':
            _, i, j = argv
            ans = 'YES' if dp.query(int(i), int(j)) else 'NO'
            print(ans)
            log.append(ans)
        elif argv[0] == 'M':
            _, i, c = argv
            dp.modify(int(i), c)
        elif argv[0] == 'P':
            _, i, k, j = argv
            ans = 'YES' if dp.pick(int(i), int(k), int(j)) else 'NO'
            print(ans)
            log.append(ans)
    return log


if __name__ == '__main__':
    # execute_test()

    for i in range(1, 16):
        print('Test {}'.format(i))
        with open('hw4/small-tests/{}.in'.format(str(i).zfill(2)), 'r') as f:
            log = execute_test(f)
        assert log == open('hw4/small-tests/{}.out'.format(str(i).zfill(2)), 'r').read().strip().split('\n')

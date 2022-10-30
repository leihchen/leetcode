
# golden_ratio = (1 + (5 ** 0.5)) / 2
#
# def fib(N: int) -> int:
#     return int(round((golden_ratio ** N) / (5 ** 0.5)))
#
# def inverse_fib(num):
#     return round(math.log(num * math.sqrt(5)) / math.log(golden_ratio))
#
# # for i in range(1, 10):
# #     print(i, ' th fib = ', fib(i))
# #     print(i, ' iloc =', inverse_fib(fib(i)))
# print(inverse_fib(10))
# print(inverse_fib(12))
#
# def twofib(num):
#     small, large = 1, num - 1
#     left, right = inverse_fib(small), inverse_fib(large)
#     # while left <= right:

def next_fib():
    a, b = 0, 1
    yield b
    while True:
        a, b = b, a + b
        yield b


def prefectSquare(num):
    x = int(math.sqrt(num))
    return x * x == num

def isFib(n):
    return prefectSquare(5 * n * n + 4) or prefectSquare(5 * n * n - 4)



def twofib(num):
    if num <= 1:
        return False
    for small in next_fib():
        if small > num / 2:
            break
        if isFib(num - small):
            print('->', small, num-small)
            return True
    return False

# for i in range(100):
#     print(i, twofib(i))


# 1 .COL.nunique()
# 2 groupby("CITY").COL.nunique()
# 3 groupby("CITY").COL.nunique()
# 4

# import seaborn as sns
# iris = sns.load_dataset('iris')
# # print(iris.species.nunique())
# # print(iris.groupby('species').sepal_length.nunique())
# #
# # sub1 = iris.groupby('species').agg(total_contract=('sepal_length', 'sum'))
# # print(sub1[sub1['total_contract'] == sub1['total_contract'].max()])
#
# sub0 = iris.groupby('species').agg(n_c=('sepal_length', 'nunique'))
# print(sub0[sub0['n_c'] > 15].head(1))


def solution(x, y):
    seen = set()
    n = len(x)
    res = 0
    test = []
    for i in range(n):
        ax, ay = x[i], y[i]
        for j in range(n):
            if i == j: continue
            bx, by = x[j], y[j]
            abx, aby = ax + bx, ay + by
            jx, jy = ay - by, bx - ax
            cx, cy = (abx + jx) / 2, (aby + jy) / 2
            print(jx, jy)
            dx, dy = (abx - jx) / 2, (aby - jy) / 2
            print([(ax, ay), (bx, by), (cx, cy), (dx, dy)])
            if (cx, cy) in seen and (dx, dy) in seen:
                test.append(([(ax, ay), (bx, by), (cx, cy), (dx, dy)]))
                # print([(ax, ay), (bx, by), (cx, cy), (dx, dy)])
                res += 1
        seen.add((ax, ay))
    for row in (test):
        print(row)
    return res
# x = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
# y = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
x = [-1, 0, -1, 0 ]
y = [0, 1, 1, 0]
plot = []
# for i in range(len(x)):
#     plot.append((x[i], y[i]))
# print(solution(x, y))
# print(plot)

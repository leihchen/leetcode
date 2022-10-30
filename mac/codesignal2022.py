from utils import *



def messager(s, n):
    MAX_DIGIT = 10
    var_digit = 0
    base_cost = 3
    ns = len(s)
    variable_cost = []
    for i in range(1, MAX_DIGIT + 1):
        var_digit += 9 * 10 ** (i - 1) * i
        variable_cost.append(var_digit)
    n_digit = -1

    for i in range(1, MAX_DIGIT + 1):
        nmsg = int('9' * i)
        total = nmsg * n
        total -= variable_cost[i-1]
        total -= (base_cost + i) * nmsg
        if total >= ns:
            n_digit = i
            break
    if n_digit == -1:
        return []
    res = []
    i = 0
    j = 1
    while i < ns:
        allowed = n - (len(str(j)) + base_cost + n_digit)
        res.append(s[i: i+allowed])
        i += allowed
        j += 1
    ans = []

    for i, segment in enumerate(res):
        ans.append('{}<{}/{}>'.format(segment, i+1, j-1))
    return ans




ans = messager("ABCD5EFGH2123495689929595995959595959", 7)
# for row in ans:
#     print(row)
ans2 = messager("ABCD EFGH", 7)
for row in ans2:
    print(row)


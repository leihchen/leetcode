# binary linked list to int
class SinglyLinkedListNode:
    def __init__(self, data=-1):
        self.data = data
        self.next = None

def from_list(list):
    dum = SinglyLinkedListNode()
    node = dum
    for i in list:
        node.next = SinglyLinkedListNode(i)
        node = node.next
    return dum.next

def getNumber(binary):
    num = 0
    while binary:
        num = num * 2 + binary.data
        binary = binary.next
    return num

# linkedlist = from_list([0,1,0,0,1,1])
# print(getNumber(linkedlist))

def alternatingPremutation(n):
    res = []
    def bt(tmp, parity, check_parity=True):
        if len(tmp) == n:
            res.append(list(tmp))
        for i in range(1, n+1):  # output should be sorted in lex
            if i in tmp or (i % 2 == parity and check_parity): continue
            tmp.append(i)
            bt(tmp, i % 2)
            tmp.pop()
    bt([], 0, False)
    return res
# alternatingPremutation(4)

def slotWheels(history):
    size = len(history)
    spin_num = len(history[0])
    map = dict()
    for i, s in enumerate(history):
        temp = [c for c in s]
        temp.sort()
        map[i] = temp
    res = 0
    for j in range(spin_num):
        cur_max = 0
        for i in range(size):
            row_max = int(map[i][-1])
            map[i] = map[i][:-1]
            cur_max = max(row_max, cur_max)
        res += cur_max
    return res
# print(slotWheels(['712', '246', '365', '312']))
import pyspark
# from pyspark.mllib.fpm import FPGrowth
# import numpy as np
# import pandas as pd
# df = pd.read_csv("census.csv", header=None)
# # transactions = buffer.map(lambda line: line.strip().split(' '))
# df = np.array(df[:10])
# print(df)
# with open('data.txt', 'r') as f:
#     buffer = f.read()
# df2 = []
# for line in buffer:
#     df2.append(line.strip().split(' '))
# print(df2)
# model = FPGrowth.train(df2, minSupport=0.6, numPartitions=1)
#
# result = model.freqItemsets().collect()
# for fi in result:
#     print(fi)
#
# with open("census.csv", 'r') as f:
#     buffer = f.read()
# print(buffer.split('\n')[:3])

from pyspark.mllib.fpm import FPGrowth
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import permutations
def attributesSet(numberOfAttributes, supportThreshold):
    # Write your code here
    res = []
    df = np.array(pd.read_csv("census.csv", header=None))
    row, col = len(df), len(df[0])
    thres = row * supportThreshold
    d = defaultdict(set)
    for i in range(row):
        for j in df[i]:
            d[j].add(i)
    dkeys = list(d.keys())
    for pattern in list(permutations(dkeys, numberOfAttributes)):
        col_name = set(pattern[0].split('=')[1])
        sz = len(pattern)
        row_set = d[pattern[0]]
        ok = True
        for i in range(1, sz):
            if pattern[i].split('=')[1] in col_name:
                ok = False
                break
            else: col_name.add(pattern[i].split('=')[1])
            row_set.union(d[pattern[i]])
        if len(row_set) >= thres and ok:
            res.append(pattern)
    print(res)
    return res

# attributesSet(2, 0.6)


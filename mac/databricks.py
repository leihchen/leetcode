from utils import *
from bisect import bisect_right
from sortedcontainers import SortedDict

# @ todo merge iterator
class LookaheadIterator:

    def __init__(self, iterable):
        self.iterator = iter(iterable)
        self.buffer = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.pop()
        else:
            return next(self.iterator)

    def hasNext(self):
        if self.buffer:
            return True

        try:
            self.buffer = [next(self.iterator)]
        except StopIteration:
            return False
        else:
            return True

# x  = LookaheadIterator(range(2))

# print(x.has_next())
# print(next(x))
# print(x.has_next())
# print(next(x))
# print(x.has_next())
# print(next(x))

class PeekingIterator:
    def __init__(self, iterator: LookaheadIterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.it = iterator
        self.top = next(self.it) if self.it.hasNext() else None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.top

    def next(self):
        """
        :rtype: int
        """
        retval = self.top
        self.top = next(self.it) if self.it.hasNext() else None
        return retval

    def hasNext(self):
        """
        :rtype: bool
        """
        return bool(self.top)

# peekingIterator = PeekingIterator(LookaheadIterator([1, 2, 3])); # // [1,2,3]
# print(peekingIterator.next());    #  // return 1, the pointer moves to the next element [1,2,3].
# print(peekingIterator.peek());    # // return 2, the pointer does not move [1,2,3].
# print(peekingIterator.next());    # // return 2, the pointer moves to the next element [1,2,3]
# print(peekingIterator.next());    # // return 3, the pointer moves to the next element [1,2,3]
# print(peekingIterator.hasNext()); # // return False

class MergingIterator:
    def __init__(self, iters: List[LookaheadIterator]):
        self.minheap = []
        for it in iters:
            pit = PeekingIterator(it)
            heappush(self.minheap, (pit.next(), pit))

    def hasNext(self):
        return len(self.minheap) != 0

    def next(self):
        element, it = heappop(self.minheap)
        if it.hasNext():
            heappush(self.minheap, (it.next(), it))
        return element

# test = list(map(LookaheadIterator, [[1,2,3], [0,4,5,6], [.5,1.5,2.5]]))
# mit = MergingIterator(test)
# while mit.hasNext():
#     print(mit.next())


# todo Revenue
class Revenue:
    def __init__(self):
        self.uid = 0
        self.id2rev = {}
        self.rev2ids = SortedDict()

    def _insert(self, rev):
        self.id2rev[self.uid] = rev
        if rev not in self.rev2ids:
            self.rev2ids[rev] = set()
        self.rev2ids[rev].add(self.uid)
        self.uid += 1
        return self.uid - 1

    def insert(self, rev, referid=-1):
        new_id = self._insert(rev)
        # update referid by rev
        if referid != -1:
            old_rev = self.id2rev[referid]
            self.rev2ids[old_rev].remove(referid)
            if old_rev + rev not in self.rev2ids:
                self.rev2ids[old_rev + rev] = set()
            self.rev2ids[old_rev + rev].add(referid)
            self.id2rev[referid] += rev
        return new_id

    def getKLowestRevenue(self, k, target):
        res = []
        idx = self.rev2ids.bisect_left(target)  # rev >= target
        for i in range(idx, len(self.rev2ids)):
            for uid in self.rev2ids[self.rev2ids.peekitem(i)[0]]:
                res.append(uid)
                if len(res) == k:
                    break
            if len(res) == k:
                break
        return res


# multiple level

# rs = Revenue()
# id1 = rs.insert(10)
# id2 = rs.insert(50, id1)
# id3 = rs.insert(20, id2)
# id4 = rs.insert(30)
# id5 = rs.insert(2)
# id6 = rs.insert(100)
# id7 = rs.insert(13, id3)
# id8 = rs.insert(23, id5)
# id9 = rs.insert(100, id7)
# print(rs.rev2ids)
# print(rs.getKLowestRevenue(2, 70))

# 60 70 33 30 25 100 113 23 100
# 0  1  2  3  4  5    6  7    8  9
# logN

# todo sql comment remove
"""
string literal
outside literal:
    '--'
    '
inside literal
    ' 
    escape char \
"""
# assume, valid escape: 1. no \ before regular char, 2. even number of \
def sqlComment(raw_str):
    source = raw_str.split('\n')
    res, buffer, open_literal = [], '', False
    for line in source:
        i = 0
        while i < len(line):
            if not open_literal and line[i] == '-' and i + 1 < len(line) and line[i+1] == '-':
                break
            else:
                buffer += line[i]
            if not open_literal and line[i] == "'":
                open_literal = True
            elif open_literal:
                if line[i] == '\\':
                    buffer += line[i+1]
                    i += 1
                elif line[i] == "'":
                    open_literal = False
            i += 1
        if buffer:
            res.append(buffer)
            buffer = ''

    return '\n'.join(res)

inp1 = r"""
SELECT * FROM Students -- here's comment1
WHERE age > 20 and name like 'A%'; -- here's '' \' \\' '\\\'' comment2
"""

inp2 = r"""
SELECT * FROM Students -- here's comment1
WHERE age > 20 and name like 'A%--not a comment'; -- here's comment2
"""

inp3 = r"""
SELECT * FROM Students -- here's comment1
WHERE age > 20 and name like 'A%--\'not a comment'; -- here's comment2
"""

inp4 = r"""
SELECT * FROM Students -- here's comment1
WHERE age > 20 and name like 'A%\\'--‍‍‍‍‌‌‌‌‍‌‍‌‍‍‍‌‍‍‍start of a comment'; -- also comment
"""

inp5 = r"""
SELECT * FROM files -- This is an inline comment
WHERE fullpath LIKE '/home/%'; -- This is an inline comment"""

inp6 = r"""
SELECT * FROM file
WHERE fullpath LIKE '--/home/%'; -- This is an inline comment"""

inp7 = r"""
SELECT * FROM files -- This is an inline comment
WHERE fullpath LIKE '\"--/home/%\"'; -- This is an inline comment"""

# print(sqlComment(inp1))
# print('---')
# print(sqlComment(inp2))
# print('---')
# print(sqlComment(inp3))
# print('---')
# print(sqlComment(inp4))
# print('---')
# print(sqlComment(inp5))
# print('---')
# print(sqlComment(inp6))
# print('---')
# print(sqlComment(inp7))


# todo O(1) tree traversal with parent ptr
class Node:
    def __init__(self, val, left=None, right=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent

root = Node(2)
n3 = Node(3, parent=root)
n9 = Node(9, parent=root)
root.left = n3
root.right = n9
n3.right = Node(32, parent=n3)
n21 = Node(21, parent=n3)
n3.left = n21
n16 = Node(16, parent=n21)
n21.right = n16
n16.left = Node(4,parent=n16)
n16.right = Node(5, parent=n16)

def preorder(root: Node):
    def next(node):
        if not node: return None
        if node.left: return node.left
        if node.right: return node.right
        p = node.parent
        while p and p.right == node:
            node = p
            p = p.parent
        if not p: return None
        return p.right

    next_node = root
    while next_node:
        print(next_node.val)
        next_node = next(next_node)

def preorder_naive(root):
    if not root:
        return
    print(root.val)
    preorder_naive(root.left)
    preorder_naive(root.right)



def inorder(root: Node):
    def next(node):
        if not node: return None
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        p = node.parent
        while p and p.right == node:
            node = p
            p = p.parent
        return p

    next_node = root
    while next_node.left:
        next_node = next_node.left
    while next_node:
        print(next_node.val)
        next_node = next(next_node)

def inorder_naive(root):
    if not root: return
    inorder_naive(root.left)
    print(root.val)
    inorder_naive(root.right)


def postorder(root):
    def next(node):
        if not node: return None
        p = node.parent
        if not p: return None
        if p.right == node:
            return p
        current = p.right
        while current and not (current.left == None and current.right == None):
            if current.left: current = current.left
            else: current.right: current = current.right
        return current
    next_node = root
    while next_node and not (next_node.left == None and next_node.right == None):
        if next_node.left: next_node = next_node.left
        else: next_node = next_node.right
    while next_node:
        print(next_node.val)
        next_node = next(next_node)

def postorder_naive(root):
    if not root: return
    postorder_naive(root.left)
    postorder_naive(root.right)
    print(root.val)

# preorder(root)
# print('----')
# preorder_naive(root)

# inorder(root)
# print('----')
# inorder_naive(root)

# postorder(root)
# print('----')
# postorder_naive(root)


# todo big string
# block list string
import llist
from llist import sllist

# class Node:
#     def __init__(self, value, next=None):
#         self.value = value
#         self.next = next
#
# class sllist:
#     def __init__(self, list):
#
#     def nodeat(self, val):
#         pass
#
#     def appendleft(self, node):
#
#
#     def pop(self):
#
#
#     def insertbefore(self, node):

class NewString:
    def __init__(self, n):
        self.index2block = {}
        self.indexlist = []
        self.blocksize = int(math.log2(n)) + 1  # why
        self.size = 0

    def find_index(self, i):
        idx = bisect_left(self.indexlist, i)
        # print(self.indexlist, i)
        if idx < len(self.indexlist) and self.indexlist[idx] == i:
            return idx
        return idx - 1

    def read(self, i):
        if i < 0 or i > self.size or len(self.index2block) == 0:
            return None
        bi = self.find_index(i)
        startIndex = self.indexlist[bi]
        sl = self.index2block[startIndex]
        return sl.nodeat(i - startIndex).value

    def insert(self, i, c):
        if i < 0 or i > self.size:
            return
        if len(self.index2block) == 0:
            sl = sllist([c])
            self.index2block[0] = sl
            self.indexlist.append(0)
        else:
            bi = self.find_index(i)
            startIndex = self.indexlist[bi]
            sl = self.index2block[startIndex]
            sl.insertbefore(c, sl.nodeat(i - startIndex))
            poppedTail = None
            for j in range(bi, len(self.indexlist)):
                blockList = self.index2block[self.indexlist[j]]
                if poppedTail != None:
                    blockList.appendleft(poppedTail)
                if j != len(self.indexlist) - 1:
                    poppedTail = blockList.pop()
            lastStartIndex = self.indexlist[-1]
            lastl = self.index2block[lastStartIndex]
            if len(lastl) > self.blocksize:
                newStartIndex = lastStartIndex + self.blocksize
                sl = sllist([])
                while len(lastl) > self.blocksize:
                    sl.appendleft(lastl.pop())
                self.index2block[newStartIndex] = sl
                self.indexlist.append(newStartIndex)
        self.size += 1

    def delete(self, i):
        if i < 0 or i > self.size or len(self.index2block) == 0:
            return
        bi = self.find_index(i)
        startIndex = self.indexlist[bi]
        sl = self.index2block[startIndex]
        sl.remove(sl.nodeat(i - startIndex))
        poppedHead = None
        for j in range(bi, len(self.indexlist))[::-1]:
            blockList = self.index2block[self.indexlist[j]]
            if poppedHead != None:
                blockList.append(poppedHead)
            if j != bi:
                poppedHead = blockList.popleft()

        lastl = self.index2block[self.indexlist[-1]]
        if len(lastl) == 0:
            j = self.indexlist.pop()
            del self.index2block[j]
        self.size -= 1



# ns = NewString(20)
# ns.insert(0, 'a');
# ns.insert(0, 'b');
# ns.insert(0, 'c');
# ns.insert(0, 'd');
# ns.insert(0, 'e');
# ns.insert(0, 'f');
# ns.insert(0, 'g');
# print(ns.index2block)
# ns.insert(0, 'h');
# ns.insert(1, 'i');
# ns.insert(2, 'j');
# ns.insert(3, 'k');
# ns.insert(4, 'l');
# ns.insert(5, 'f');
# ns.insert(6, 'm');
# ns.insert(7, 'n');
# ns.insert(12, 'x');
# print(ns.index2block)
# print(ns.read(3))
# print(ns.read(5))
# print(ns.read(12))
# ns.delete(12);
# print(ns.index2block)
# ns.delete(1);
# print(ns.index2block)
# ns.delete(2);
# print(ns.index2block)
# ns.delete(0);
# print(ns.index2block)
# ns.delete(0);
# ns.delete(0);
# ns.delete(0);
# print(ns.index2block)

# todo snapshot set
# history log for each values
# append only log for value tracking, we can't remove it,
# copy on write record,
# edit within snapshot boundary
# copy when accessed across boundary

class SnapshotIterator:
    def __init__(self, snapshot_set, snapshot_size, snapshot_id):
        self.snapshot_set = snapshot_set
        self.snapshot_size = snapshot_size
        self.snapshot_id = snapshot_id

        self.i = 0
        self.buffer = self.peek()

    def has_next(self):
        return bool(self.buffer)

    def peek(self):
        while self.i < self.snapshot_size and \
                not self.snapshot_set.contains(self.snapshot_set.datalist[self.i], self.snapshot_id):
            self.i += 1
        if self.i == self.snapshot_size:
            return None
        return self.snapshot_set.datalist[self.i]

    def next(self):
        result = self.buffer
        self.i += 1
        self.buffer = self.peek()
        return result

class SnapshotSet:
    def __init__(self):
        self.datalist = []
        self.snap_id = 0
        self.snapshots = defaultdict(list)

    def add(self, e) -> bool:
        if e not in self.snapshots:
            self.snapshots[e].append([self.snap_id, 1])
            self.datalist.append(e)
            return True
        exists = self.snapshots[e][-1][1] == 1
        if self.snapshots[e][-1][0] == self.snap_id:
            self.snapshots[e][-1][1] = 1
        else:
            self.snapshots[e].append([self.snap_id, 1])
        return not exists

    def remove(self, e) -> bool:
        if e not in self.snapshots:
            return False
        exists = self.snapshots[e][-1][1] == 1
        if self.snapshots[e][-1][0] == self.snap_id:
            self.snapshots[e][-1][1] = 0
        else:
            self.snapshots[e].append([self.snap_id, 0])
        return exists

    def contains(self, e, snap_id=-1):
        if snap_id == -1:
            return self.snapshots[e][-1][1] == 1
        if snap_id > self.snap_id:
            raise IndexError
        if e not in self.snapshots:
            return False
        this_snapshot = self.snapshots[e]
        idx = bisect_left(this_snapshot, [snap_id, -1])
        if idx == len(this_snapshot):
            return self.snapshots[e][-1][1] == 1
        if this_snapshot[idx][0] == snap_id:
            return this_snapshot[idx][1] == 1
        if idx - 1 < 0:  # request snap_id is smaller than first add
            return False
        return this_snapshot[idx - 1][1] == 1

    def iterator(self) -> SnapshotIterator:
        result = SnapshotIterator(self, len(self.datalist), self.snap_id)
        self.snap_id += 1
        return result

# ss = SnapshotSet()
# ss.add('k1')
# ss.add('k2')
# ss.add('k3')
#
# it1 = ss.iterator()
# ss.add('k4')
# ss.remove('k3')
#
# print('it1:')
# # k1, k2, k3
# while it1.has_next():
#     print(it1.next())
#
# it2 = ss.iterator()
# print('it2:')
# # k1, k2, k4
# while it2.has_next():
#     print(it2.next())
#
# assert ss.contains('k1')
# assert ss.contains('k4')
# assert not ss.contains('k3')
#
# ss.add('k1')
# ss.add('k5')
# ss.add('k3')
# it3 = ss.iterator()
# print('it3:')
# # k1, k2, k3, k4, k5
# while it3.has_next():
#     print(it3.next())
#
# ss.remove('k1')
# assert not ss.contains('k1')
# ss.add('k1')
# assert ss.contains('k1')
#
# ss.remove('k1')
# it4 = ss.iterator()
# ss.add('k1')
#
# print('it4:')
# print(ss.datalist, ss.snapshots)
# # k2, k3, k4, k5
# while it4.has_next():
#     print(it4.next())

import time
# todo mockhashmap
# 如果允许一定的误差, 能否优化时间空间复杂度
class MeasureQPS:

    def get(self, key):
        pass
    def put(self, key, val):
        pass
    def measure_get(self):
        pass
    def measure_put(self):
        pass

# todo mutlithread web crawler
# follow up how to work in distributed setup
# 问的比较细，各种disk io，network io优化
print(time.time())

# top k frequent in stream, with map reduce



# Round 5   Design WebCrawler
# First write single threaded Web Crawler, Then write multi-threaded WebCrawler.
# // 一开始提供了三个utility function：
# // (1）fetch(URL: url) -> HTML: content;
# // （2）parse(HTML: content) -> List<URL>;
# // (3) save(URL: url, HTML: content)。
# // `save`是把数据存在disk上，`fetch`是发个network req‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌uest，
# // `parse`是in-memory解析html page，
# /**
# * improve the efficiency by multi-thread
# * keys
# * 1 - avoid race condition where 2 threads check same url
# * and try to process as next url at same time (this causes duplicate visit on same url)
# * use concurrentHashMap put to avoid, since the put (insert entry) lock the segment of map
# * and if return null meaning no such key in map previously which means we can process the url
# * 2 - save is a disk I/O where we should put it into a separate thread pool to let it finish by itself
# * 3 - fetch html is a network I/O
# *
# */

import functools
# todo lazy array
class LazyArray:
    def __init__(self, data: Iterable):
        self.data = data
        self.funcList = []
        self.prefixFunc = lambda x: x

    def map(self, f):
        self.funcList.append(f)
        self.prefixFunc = functools.reduce(lambda f, g: lambda x: f(g(x)), [f, self.prefixFunc])

        return self

    def get(self):
        res = self.data
        for f in self.funcList:
            res = map(f, res)
        return res

    def index(self, x):
        for i, d in enumerate(self.data):
            if self.prefixFunc(d) == x:
                return i
        return -1

lz = LazyArray([10, 20, 30, 40, 50]).map(lambda x: x**2).map(lambda x:x*3).map(lambda x: (x,x)).index((2700, 2700))
# print((lz))

def cidr_match(ip, cidr):
    ruleip, mask = cidr.split('/')
    mask = int(mask)
    def ip2number(ip):
        numbers = list(map(int, ip.split('.')))
        return (numbers[0] << 24) + (numbers[1] << 16) + (numbers[2] << 8) + (numbers[3])
    src = ip2number(ip)
    dst = ip2number(ruleip)
    return src >> (32 - mask) == dst >> (32 - mask)

# print(cidr_match('255.0.0.12', '255.0.0.8/29'))
# print(cidr_match('255.0.0.7', '255.0.0.8/29'))

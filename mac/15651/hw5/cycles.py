from collections import deque, defaultdict, Counter
from copy import deepcopy
import sys
class Graph:
    def __init__(self,):
        self.adjlist = defaultdict(Counter)

    def add_edge(self, u, v, capacity=0):
        self.adjlist[u][v] = capacity

    def get_edges(self, w=False):
        if w:
            return [(u, v, w) for u in self.adjlist for v, w in self.adjlist[u].items()]
        else:
            return [(u, v, w) for u in self.adjlist for v, w in self.adjlist[u].items() if w > 0]

    def add_node(self, u):
        pass

    def has_edge(self, u, v):
        return self.adjlist[u][v] > 0

    def has_node(self, u,):
        return self.adjlist[u] != None


def Ford_Fulkerson(graph, source, sink):
    """Returns the maximum flow from source to sink in graph, update graph to residual graph"""
    residual = deepcopy(graph)
    g = residual.adjlist
    def BFS(g, source, sink):
        queue = deque([source])
        parent = {source: None}
        while queue:
            u = queue.popleft()
            for v in g[u]:
                if v not in parent and g[u][v] > 0:
                    queue.append(v)
                    parent[v] = u
        if sink not in parent:
            return None
        path = []
        while sink is not source:
            path.append((parent[sink], sink))
            sink = parent[sink]
        return path

    flow = 0
    while True:
        path = BFS(g, source, sink)
        if path is None:
            return flow, residual
        min_cap = min(g[u][v] for u, v in path)
        flow += min_cap
        for u, v in path:
            g[u][v] -= min_cap
            g[v][u] += min_cap

dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
def solution(grid):
    G = Graph()
    n, m = len(grid), len(grid[0])

    def add_nswe(grid, i, j):
        for dx, dy in dirs:
            newx, newy = i + dx, j + dy
            if 0 <= newx < n and 0 <= newy < m and grid[newx][newy] != 'x':
                u = (i, j)
                v = (newx, newy)
                if not G.has_node(u):
                    G.add_node(u)
                    G.add_node(v)
                if (i + j) % 2 == 0:
                    G.add_edge((i, j), (newx, newy), capacity=1)
                else:
                    G.add_edge((newx, newy), (i, j), capacity=1)

    for i in range(n):
        for j in range(m):
            u = (i, j)
            if grid[i][j] == '.':
                if not G.has_node(u):
                    G.add_node(u)
                if (i + j) % 2 == 0:
                    G.add_edge('s', (i, j), capacity=2)
                    add_nswe(grid, i, j)
                else:
                    G.add_edge((i, j), 't', capacity=2)
                    add_nswe(grid, i, j)
    maxflow, residual = Ford_Fulkerson(G, 's', 't')
    udg = defaultdict(list)
    for u, v, w in residual.get_edges(w=False):
        if u != 's' and v != 't' and u != 't' and v != 's':
            if G.has_edge(u, v):
                continue
            udg[u].append(v)
            udg[v].append(u)
    if maxflow != sum(row.count('.') for row in grid):
        print('NO')
        return
    print('YES')
    visited = set()
    res = [list(row) for row in grid]

    visited = set()
    def dfs(i, j):
        if (i, j) in visited:
            return
        visited.add((i, j))
        res[i][j] = '+'
        a, b = udg[(i, j)]
        if a[0] == b[0] == i:
            res[i][j] = '-'
        elif a[1] == b[1] == j:
            res[i][j] = '|'
        elif a[0] + a[1] == b[0] + b[1]:
            if i + j > a[0] + a[1]:
                res[i][j] = 'J'
            else:
                res[i][j] = 'r'
        elif a[0] - a[1] == b[0] - b[1]:
            if i - j > a[0] - a[1]:
                res[i][j] = 'L'
            else:
                res[i][j] = '7'
        for nei in udg[(i, j)]:
            if nei not in visited:
                dfs(nei[0], nei[1])


    for i in range(n):
        for j in range(m):
            if grid[i][j] == '.':
                dfs(i, j)
    for row in res:
        print(''.join(row))

grid = []
nrow, _ = sys.stdin.readline().strip().split()
for _ in range(int(nrow)):
    grid.append(sys.stdin.readline().strip())
solution(grid)
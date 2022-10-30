from utils import *

def book(endings=[5,10], choices=[[3,5,9], [9,8,10]]):
    page2choices = {c[0]: (c[1], c[2]) for c in choices}
    endingcnt = {ending: 0 for ending in endings}
    m = max(endings)
    visited = [False] * (m+1)
    def bt(start, visited):
        if start in endingcnt:
            endingcnt[start] += 1
            return
        if start in page2choices:
            for nextpage in page2choices[start]:
                if not visited[nextpage]:
                    visited[nextpage] = True
                    bt(nextpage, visited)
                    visited[nextpage] = False
        else:
            visited[start + 1] = True
            bt(start + 1, visited)
            visited[start + 1] = False
    bt(1, visited)
    return endingcnt


# print(book())


badge_records = [
    ["Paul", "1214", "enter"],
["Paul", "830", "enter"],
["Curtis", "1100", "enter"],
["Paul", "903", "exit"],
["John", "908", "exit"],
["Paul", "1235", "exit"],
["Jennifer", "900", "exit"],
["Curtis", "1330", "exit"],
["John", "815", "enter"],
["Jennifer", "1217", "enter"],
["Curtis", "745", "enter"],
["John", "1230", "enter"],
["Jennifer","800", "enter"],
["John", "1235", "exit"],
["Curtis", "810", "exit"],
["Jennifer", "1240", "exit"],
]
for row in sorted(badge_records, key = lambda x: int(x[1])):
    print(row)


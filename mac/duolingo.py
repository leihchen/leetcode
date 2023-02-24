from utils import *


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.isWord = False


class Trie:
    def __init__(self, words):
        self.root = TrieNode()
        for word in words:
            self.insert(word)

    def insert(self, word):
        node = self.root
        for c in word:
            node = node.children[c]
        node.isWord = True

class Cipher:
    def __init__(self, dict):
        self.encode_map = dict
        self.decode_map = defaultdict(list)
        for k, v in dict.items():
            self.decode_map[v].append(k)

    def encode(self, plain):
        res = []
        for c in plain:
            res.append(self.encode_map[c])
        return ''.join(res)

    def decode(self, cipher, vocab):
        res = []
        trie = Trie(vocab)
        def dfs(tmp: List[str], i, last_node: TrieNode):
            nonlocal res
            if i == len(cipher) // 2:
                res.append(''.join(tmp))
                return
            for candidate in self.decode_map[cipher[i * 2:i * 2 + 2]]:
                if candidate in last_node.children:
                    dfs(tmp + [candidate], i + 1, last_node.children[candidate])

        dfs([], 0, trie.root)
        return res



# codex = Cipher({'z': 'aw', 'r': 'wo', 't': 'wo'})
# print(codex.decode_map)
# print(codex.decode('wowo', ['rt', 'tr', 'rr', 'tt']))

from collections import namedtuple
Streak = namedtuple('Streak', ('length', 'start_date', 'end_date'))

from datetime import datetime

def streak_dates(dates):
    parsed = set([datetime.strptime(date, '%Y-%m-%d').toordinal() for date in dates])
    res = []
    for date in parsed:
        if date - 1 not in parsed:
            y = date + 1
            while y in parsed:
                y += 1
            res.append(Streak(y - date, datetime.strftime(datetime.fromordinal(date), '%Y-%m-%d'), datetime.strftime(datetime.fromordinal(y-1), '%Y-%m-%d')))
    return res

# print(streak_dates(["2021-01-16", "2021-01-17", "2021-01-15", "2021-01-10"]))


orders = ["sushi", "sushi", "chicken fried rice", "sushi"]
recipes = { "sushi":  ["salmon", "rice"], "chicken fried rice": ["chicken", "fried rice"], "fried rice": ["egg", "onion", "rice"] }
prep_times = { "salmon": 1, "rice": 2, "egg": 1, "onion": 1, "chicken": 2 }
cooldown = 3

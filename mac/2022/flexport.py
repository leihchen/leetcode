# # def intersect(left, right):
# #     i = 0
# #     for x in left:
# #         for i in range(i, len(right)):
# #             if right[i] > x:
# #                 break
# #             if right[i] == x:
# #                 yield x
# #
# # left = [1,2,3,4,5,6,7,7]
# # right = [2,4,5,5,7]
# # for v in intersect(left, right):
# #     print(v)
# def f(n):
#     if n <= 0:
#         return 0
#     return n + f(int(n/2))
# print(f(4))
#
# testvec = [[1,1,3,4], [2,2], [4,3,3,1], [6,3,2,4,3,1], [10,2,3,4,5,6,7,8,9,10]]
# goldvec = [3,3,5,8,11]
#
# def find_mistake(nums):
#     n = len(nums)
#     count = [0] * n
#     for num in nums:
#         count[num-1] += 1
#     res = 0
#     for i, cnt in enumerate(count):
#         if cnt != 1:
#             res += i + 1
#     return res
#
# def case(func, tvec, gvec):
#     for i in range(len(tvec)):
#         print(func(tvec[i]), gvec[i], func(tvec[i]) == gvec[i])
#
# # case(find_mistake, testvec, goldvec)
#
# def almost_palindrome(str):
#     res = 0
#     left, right = 0, len(str)-1
#     while left < right:
#         res += 2 if str[left] != str[right] else 0
#         left += 1
#         right -= 1
#     return res
#
# t = ['abba', 'racecar', 'abcdcaa', 'aaabbb', 'abcdefgh', 'a']
# g = [0,0,2,6,8,0]
# case(almost_palindrome, t, g)
#
# def can_reach_end(nums):
#     far = nums[0]
#     for i in range(1, len(nums)):
#         if far < i:
#             return False
#         far = max(far, i + nums[i])
#     return far >= len(nums) - 1
#
# t3 = [[1,2,3], [5,0,0,0], [0], [0, 2,4], [1,2,0,0,1]]
# g3 = [1, 1, 1, 0, 0]
# case(can_reach_end, t3, g3)
import queue, threading, time
# q = queue.Queue()
# for i in [3,2,1]:
#     def f():
#         time.sleep(i)
#         q.put(i)
#     threading.Thread(target=f).start()
# print(q.get())

# def f(sl):
#     root = {}
#     for s in sl:
#         base = root
#         for word in s.split(' '):
#             if not base.get(word):
#                 base[word] = {}
#             base = base[word]
#     return root
# tree = f(['hello word', 'hello there'])
# print(tree)

# def func(a,b):
#     a += 1
#     b.append(1)
# a,b = 0 , []
# func(a, b)
# print(a,b)
#
# def f(items):
#     i = 0
#     while i < len(items):
#         if len(items[i]) == 0:
#             del items[i]
#         i += 1
# names = ['r', '', 'm', '', '', 't']
# f(names)
# print(names)

# f = lambda n : 1 if n <= 1 else n * f(n-1)
# print(f(4))
# words = ['Hello', 'World']
# for i, word in enumerate(words):
#     word = word.lower()
#     words[i] = word[::-1]
# print(words)
# def f(nums):
#     hp = hn = False
#     for num in nums:
#         hp = num > 0
#         hn = num < 0
#     return (hp, hn)
# print(f([-1, 0, 1]))

# def f(func, items):
#     i = 0
#     for item in items:
#         if func(item):
#             items[i] = item
#             i += 1
#     del items[i:]
# func = lambda x : x % 2 == 1
# l = [1,2,3,4,5,6,7]
# print(f(func, l))
# print(l)
try:
    file = open('sfs.txt')
    data = file.read()
finally:
    file.close()
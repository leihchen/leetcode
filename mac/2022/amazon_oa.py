import utils

# 亚麻SDE 1 OA
# 两道题 105 分钟 7分钟 work simulation
# 1. Amazom warehouse 相关的题 是 lc药思散罢 的变种
# 2.Economy Mart 相关的‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌题 lc  饵以岭饵变种
# 求米
# https://www.1point3acres.com/bbs/thread-908750-1-1.html
# 两个逻辑直白的题但是总会有超时的case。。。还是水平不够
# 1. 有很多仓里面有包裹，每次对每个仓储只能处理同样数量的包裹，除了0个都不能有负，几次可以处理完。
# 2. 有很多站点，对每个连续子集的一‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌个数据是min*sum， 求这个数据总的数据值
# 攒攒人品，回去继续努力。
# 1. 具體內容不太記得了，簡單說給一個int array，問要幾個swap (只能相鄰的swap) 使得最小的數在第一個 最大的數在最後一個for example,
# given [2,3,1,5,4], return 3. [2,3,1,5,4] -> [2,1,3,5,4] -> [1,2,3,5,4] -> [1,2,3,4,5]
# 2. https://leetcode.com/discuss/int ... need-help-amazon-oa
# 第一题，pascal triangle的变形：给一个integer array，一直做pascal triangle reduce，直到array size == 2，并且每一层只保留last digit。
# e.g. input: [2 8 6 7 1], [2 8 6 7 1] -> [0 4 3 8] -> [4 7 1] -> [1 8]，return [1 8]
# 第二题parcel and truck，参考答案https://www.1point3acres.com/bbs ... 8205;‌-1.html，刷题网 而要就无 变形题
# 第一题brute force O(n ^ 2)，第二题做了一个hashset然后while loop，O(n)
# 面试题目：
# 1. 这道不难，无重复的数组把最小的移到最左边，最大的移到最右边，每次只能swap相邻的两个，求最少的swap次数。
# 解法可以看这个帖子
# https://www.1point3acres.com/bbs/thread-886248-1-1.html
# 2. 这道有点难度，之前地里看过好多帖子说这个。有点像里扣巴尔巴和尔尔留尔。
# 以下内容需要积分高于 120 您已经可以浏览
# 具体内容和解法大家可以看这个帖子
# https://leetcode.com/discuss/interview-question/1802061/Amazon-SDE2-OA
# 仔细读题，因为有一个坑点，如果一个数组是 1，3，5，那 imbalance 是算 2 不是 1。
# work simulation 更像是技术栈的考察。问题大概是现在有一个工作，还没开始，然后下面五个选项，比如确定用户需求，设计测试用例，让你选 effective 的程度。还有就是现在有一个需求，比如投票系统，短时间高并发，还要能编辑，能在投票结束后随时回看结果。底下会有一些不同的数据库，让你选这些数据库对这个问题有用的程度。
# work style 给我感觉像是 LP + 性格测试，靠着之前准备的 LP 硬着头皮做完。比较无语的是有一些题目是二选一哪个更像我：同事总是指出我的问题还是我工作中经常犯错。
# 这种死亡二选一我真的会崩溃。


# 315 Count of Smaller Numbers After Self / Algorithm swap
# 493 reverse pairs
# 829 consecutive numbers sum
# 994 rotting orange
# 1043. Partition Array for Maximum Sum
# 1335 Minimum Difficulty of a Job Schedule
# 21: Merge two sorted Lists
# 200. Number of Islands
# 323. Number of Connected Components in an Undirected Graph / Cloudfront Caching
# 414. Third Maximum Number
# 547: number of province
# 692: top K frequent word
# 937: reorder data in log files
# 973: K closest point
# 1010. Pairs of Songs With Total Durations Divisible by 60 / Amazon Music
# 1041: robot bounded in circle
# 1099. Two Sum Less Than K
# 1135. Connecting Cities With Minimum Cost (MST)
# 1167. Minimum Cost to Connect Sticks
# 1197. Minimum Knight Moves / Demolition Robot
# 1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts / Storage Optimize
# 1629: Slowest key
# 1710. Maximum Units on a Truck
# ?: split into two
# ?: Optimize Box Weight
# ?: Device Application Pairs / Optimal Amazon Air Route / Optimal Utilization
# ?: Shopping Option / shopping selection
# ?: Find All Combination of Numbers that sum to a target
# ?: Five star seller
# ?: Split string into unique primes
# ?: Event Scheduler
# ?: Longest K-interspace Substring
# ?: Split Parentheses
# ?: Two sum unique pairs
# ?: Secret Array
# ?: Autoscale Policy, Utilization Check
# ?: Decode String
# ?: different ways to choose fleets of vehicle
# OA 都是地里的原题，非常感恩
# 1. package merge weight 求最大值
# 2. no more than k segment （count shipment imbalance）
# 下面是一写总结的不完全统计
# SDEII OA
# 1.list头尾sum *4
# 2.dispatch parcel 数非零去重 *5
# 3.pile *4
# 4.max deviation *5
# 5.min swap 01回文 *3
# 6.连续下降评分 *2
# 7.no more than k segment （count shipment imbalance）*3
# 8.子数组差之和2104原题 *4
# 9.channel中位数 *2
# 10. veg restaurant *1
# 11. mix cluster memory front back
# 12. package merge weight 求最大值*1
# 13. power * 2
# 14. max length of valid server cluster * 1
# 15. max subarray length with product 1 * 2
# 16. 10移到一边 * 2
# 17.密码强度 不同字符和 *2
# 18. 无重复的数组把最小的移到左边最大的移到右边只能swap相邻的两个。求最少的swap
# 19.Insert ship
# 20.DNA 弄palindrome detection
# 21.3 pages
# 22.MEX
# 23.尔尔留尔
# 24.整体右移 match count
# 25.Min steps make ‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌array non decreasing
# 第一题是给一个array和数字k，求删除k个连续的元素后array和的最小值。
# example: arr = [7,3,6,1], k=3, result=7
# explain: delete [3,6,1] from the array
# 第二题是给一个array代表movie的数量和一个数字k，如果一个subarray里的最大值减最小值小于k，这个subarray就是一个group，题目要求找出group数量的最小值。
# example: movies = [1,5,4,6,8,9,2], k=3, result=3
# explain: movie‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌s can be divided to 3 groups: [2,1],[5,4,6],[8,9]
# 第一题滑动窗口，第二题brute force，都pass all test cases了，一小时收到OA2，还没开始做。
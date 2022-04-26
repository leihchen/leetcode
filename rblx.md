Graph: MST, SSSP

parallel bfs topological sort, many machine





1. Single-tab browser

    ```python
    
    ```

    

2. Random Pick with Weight https://leetcode.com/problems/random-pick-with-weight/

    ```python
    i  0, 1, 2
    w [1, 2, 3]  total_weight = 6
    1/6 chance to pick 0
    2/6 chance to pick 1
    3/6 chance to pick 2
    
    roll a fair six face die
     1 2 3 4 5 6
    [0,1,1,2,2,2]
    
    1. lookup solution O(1), O(w[i])
     0   1   2 
    [w1, w2, w3]
    concate([duplicate(0, w1), duplicate(1, w2), duplicate(2, w3)])
    
    2. binary-search solution O(lgN), O(N)
     1|2 3|4 5 6     -> 1|3|6 accum weight
    [0|1,1|2,2,2]  			0|1|2
    ex: rand = 4, find the next non-strictly larger number 
    ```

3. word search https://leetcode.com/problems/word-search/

    ```python
    backtracking O(N^2 * 4^L) where N= number of cells, L=len(word)
    (dfs O(N^2), backtracking choice^pathLen)
    ```

4. peak index https://leetcode.com/problems/find-peak-element/

    ```python
    # 853, yhc's note 
    	  V
        - 
       -
         -
     -
    -     -
    0123456  idx
       ^
       i
    0001111  f(idx)
    f(i) = A[i] > A[i+1]
    # 162, not given moutain shape, nums[i] != nums[i+1]
    same
    
    
    ```

5. course schedule II https://leetcode.com/problems/course-schedule-ii/

    ```python
    cycle detection problem in a graph (to be more specific, it is a directed graph)
    the nodes are courses, edges are dependences
    
    101 -> 201; 102 -> 201; 105 -> 205
    201 -> 301; 205 -> 301
    keep track of a number of prequites of each course
    
    Can take <- [102, 105], pop from the can take queue once a time and add it to taken 
    Taken [101,102,] <-
    
    
    # 1. convert edge list to adjacency list
    # 2. find src node by calculating indegree (pre-count) of each node
    # 3. do bfs as shown above
    
    # what about cycle
    # if taken len(course) > len(taken), there exists some course that never finish
    
    1. O(E)  O(E) 
    2. O(E)  O(V)
    3. bfs: visit each node exactly one time, every time move it from cantake to taken O(V+E) 
    ```

6. 206 reserved linked list https://leetcode.com/problems/reverse-linked-list/

    ```python
    iteravitve: 
    1. grab a pointer to the next
    NULL	1 ---->2---->3->4->5->NULL
    prev trav   next
    2. rewire
    NULL<-1			 2---->
    prev trav   next
    3. advance
    NULL<-1<-----2---->3->4
    		 prev		trav  next
    4. finish off at tail of list
    NULL<-1<-2<-3<-4<-----5    NULL
    							prev   trav  next
    NULL<-1<-2<-3<-4<-----5    NULL
    							       prev  trav (finish when trav is NULL, return prev)
    
    recursive:
    1 ---> 2 ---> ... ---> 5 ---> NULL
    
    base case:
    	if i have no next, i will return my self
    work:
      1.duty of myself, to ask my next to point at me
      node.next->next = node;
      2. ask my next to do the same, this happens before 1.
    	3. i need to free up my next to null
    res = f(node.next)
    node->next->next = node
    return res
    
    ```

7. continuous subarray sum

    ```python
    find i and j such that sum(arr[i:j]) % k = 0
    
    accum[i] = sum(arr[0:i])
    then subarray sum [i:j] = accum[j] - accum[i]
    diff you two number is multiple of k: remainder is the same
    
    r[j] = sum(arr[0:j]) % k
    every time, we check if there're previous r[i] with r[i] == r[j]
    
    ```

8. subarray sum equal k

    ```python
    1. brute force n^3
    2. brute force n^2
    accum[i] = sum(arr[0:i])
    sum of subarray [i, j] = accum[j] - accum[i-1]
    this generates all the (i,j) pairs, we don't need that
    3. optimal n
    <accum[i], count(i)> 
    count(i) such that i < j and  accum[j] - accum[i] = k
    count(i) = number of subarrys sum to k which ends at location j
    ```

9. 1197 knight

    ```python
    obstacles what's heuristic
    ```

10. combination sum

636

top k quick select

```
We want to know what the Top Game is, defined by: The Top Game is the game users spent the most time in.
Each line of the file has the following information (comma separated):
- timestamp in seconds (long)
- user id (string)
- game id (int)
- action (string, either "join" or "quit")
e.g.
[
"1000000000,user1,1001,join", // user 1 joined game 1001
"1000000005,user2,1002,join", // user 2 joined game 1002
"1000000010,user1,1001,quit", // user 1 quit game 1001 after 10 seconds
"1000000020,user2,1002,quit", // user 2 quit game 1002 after 15 seconds
];
In this log,
The total time spent in game 1001 is 10 seconds.
The total time spent in game 1002 is 15 seconds.
Hence game 1002 is the Top Game. -> 1002
This file could be missing some lines of data (e.g. the user joined, but then the app crashed).
If data for a session (join to quit) is incomplete, please discard the session.
To recover some data, we attempt to estimate session length with this rule:
  If a user joined a game but did not leave, assume they spent the minimum of
    - time spent before they joined a different game; and
    - average time spent across the same user's gaming sessions (from join to leave)
    e.g.
    "1500000000,user1,1001,join"
    "1500000010,user1,1002,join"
    "1500000015,user1,1002,quit"
    The user spent 5 seconds in game 2, so we assume they spent 5 seconds in game 1.
Write a function that returns the top game ID, given an array of strings representing
each line in the log file.
```



valid sudoku



多个tab

heuristic obstacled knight jump

peak finding problem

elevator problem

OOP longest time player

OOP Connect Four problem  class supporting drop(col), check(row, col) APIs.




open-end problem: (经典system design的开放问题)

Give you Three Elevators for a 10-floor office building， how would you design the system such that all/most of the people in the office building are happy

楼主提出了5点possible idea,包括: 让elevators尽可能隔开， improve interior 和 exterior， provide a person to push the buttons for you etc.

如何了解用户对system的满意程度

There are many criteria we could consider: the number of passengers delivered to their destination, the time a passenger waits for an elevator to arrive, the time spent inside the elevator, the cost of the power consumed by the system, the wear and tear from moving and reversing directions, etc (or the resulting time and money spent on maintenance).

Fast(min time in elevator, min waiting time)

Interactiveness (display showing current location on each floor, predictable behaviour)

design parking lot

design match making system

該怎麽降低 1. 小朋友偷家長信用卡充值， 2. 某些人偷別人信用卡充值





BQ:

不在简历上最精彩的经历

最骄傲的事

最后悔的事

what would you do if you gonna die

If you live each day as if it was your last, someday you’ll most certainly be right.

dead soon is the most important tool I’ve ever encountered to help me make the big choices in life

Because almost everything — all external expectations, all pride, all fear of embarrassment or failure — these things just fall away in the face of death, leaving only what is truly important. There is no reason not to follow your heart.

what's the hardest decision you ever made

one experience working with other people

one experience learning something in a unstructured way

what's the most proud project you have ever worked on?

what's the hardest decision I have made.

最糟糕的team work經歷

的最难的课和不同寻常的经历

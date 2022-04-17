Graph: MST, SSSP

parallel bfs topological sort, many machine





1. Single-tab browser

    ```python
    
    ```

    

2. Random Pick with Weight

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

3. word search

    ```python
    backtracking O(N^2 * 4^L) where N= number of cells, L=len(word)
    (dfs O(N^2), backtracking choice^pathLen)
    ```

4. peak index

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

5. course schedule II

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

6. 206 reserved linked list

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

10. 





多个tab

heuristic obstacled knight jump

peak finding problem

elevator problem





open-end problem:

improve elevator  

package com.company;

import java.util.*;

public class Syntax {
    // monotonic stack
    private static int[] nextGeaterElement(int[] nums) {
        int[] res = new int[nums.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && nums[i] >= stack.peek()) {
                stack.pop();
            }
            res[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(nums[i]);
        }
        return res;
    }

    // quick select

    // sliding window
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        Map<Character, Integer> map = new HashMap<>();
        int left = 0, res = 0;
        for (int i = 0; i < s.length(); i++) {
            char cur = s.charAt(i);
            map.put(cur, map.getOrDefault(cur, 0) + 1);
            while (map.size() > k) {
                char c = s.charAt(left);
                map.put(c, map.get(c) - 1);
                if (map.get(c) == 0) map.remove(c);
                left++;
            }
            res = Math.max(res, i - left + 1);
        }
        return res;
    }
    // sort
    // Arrays.sort(nums);
    // PQ
//     PriorityQueue<Integer> pq = new PriorityQueue<>((a,b) -> a - b);
    // pq.offer();
    // pq.poll();

    // topological sort
    public boolean canFinish(int N, int[][] edges) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        int[] indegree = new int[N];
        for (int[] edge : edges) {
            int end = edge[0], start = edge[1];
            graph.computeIfAbsent(start, x -> new ArrayList<>()).add(end);
            indegree[end]++;
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < N; i++)
            if (indegree[i] == 0) q.add(i);
        int count = 0;
        while (!q.isEmpty()) {
            int cur = q.poll();
            count++;
            for (int nei : graph.getOrDefault(cur, new ArrayList<>()))
                if (--indegree[nei] == 0) q.offer(nei);
        }
        return count == N;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    private void dfs(TreeNode node, Map<TreeNode, List<TreeNode>> graph) {
        if (node.left != null) {
            graph.getOrDefault(node.left, new ArrayList<>()).add(node);
            graph.getOrDefault(node, new ArrayList<>()).add(node);
            dfs(node.left, graph);
        }
        if (node.right != null) {
            graph.getOrDefault(node.right, new ArrayList<>()).add(node);
            graph.getOrDefault(node, new ArrayList<>()).add(node.right);
            dfs(node.right, graph);
        }
    }

    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        Map<TreeNode, List<TreeNode>> graph = new HashMap<>();
        dfs(root, graph);
        Queue<TreeNode> q = new LinkedList<>();
        q.add(target);
        Set<TreeNode> visited = new HashSet<>();
        for (int i = 0; i < k; i++) {
            int sz = q.size();
            for (int j = 0; j < sz; j++) {
                TreeNode node = q.poll();
                for (TreeNode nei : graph.get(node)) {
                    if (visited.add(nei)) q.add(nei);
                }
            }
        }
        List<Integer> res = new ArrayList<>();
        for (TreeNode node : q) {
            res.add(node.val);
        }
        return res;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }
    // examples
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        for (ListNode head : lists) {
            pq.add(head);
        }
        while (!pq.isEmpty()) {
            ListNode node = pq.poll();
            if (node.next != null) {
                pq.add(node.next);
            }
            cur.next = node;
            cur = cur.next;
        }
        return dummy.next;
    }

    public String alienOrder(String[] words) {
        Map<Character, Integer> indegree = new HashMap<>();
        for (String s : words) {
            for (Character c : s.toCharArray()) {
                indegree.put(c, 0);
            }
        }
        Map<Character, List<Character>> graph = new HashMap<>();
        for (int i = 1; i < words.length; i++) {
            String word1 = words[i - 1], word2 = words[i];
            if (word1 == word2) continue;
            if (word1.length() > word2.length() && word1.startsWith(word2)) return "";
            for (int j = 0; j < Math.min(word1.length(), word2.length()); j++) {
                if (word1.charAt(j) != word2.charAt(j)) {
                    indegree.put(word2.charAt(j), indegree.getOrDefault(word2.charAt(j), 0) + 1);
                    graph.computeIfAbsent(word1.charAt(j), x -> new ArrayList<>()).add(word2.charAt(j));
                    break;
                }
            }
        }
        Queue<Character> q = new LinkedList<>();
        StringBuilder sb = new StringBuilder();
        for (Character key : indegree.keySet()) {
            if (indegree.get(key) == 0) {
                q.add(key);
                sb.append(key);
            }
        }

        while (!q.isEmpty()) {
            Character c = q.poll();
            for (Character nei : graph.getOrDefault(c, new ArrayList<>())) {
                indegree.put(nei, indegree.get(nei) + 1);
                if (indegree.get(nei) == 0) {
                    q.add(nei);
                    sb.append(nei);
                }
            }
        }
        if (sb.length() != indegree.size()) return "";
        return sb.toString();
    }

    public int lengthOfLIS(int[] nums) {
        List<Integer> lis = new ArrayList<>();
        for (int num: nums){
            int idx = Collections.binarySearch(lis, num);
            if (idx < 0) lis.add(num);
            else lis.set(idx, num);
        }
        return lis.size();
    }
    static class MyHashMap<K, V> {

        final int size = 10000;
        class Node<K,V>{
            final K key;
            V value;
            public Node(K key, V value){
                this.key = key;
                this.value = value;
            }
            public K getKey(){
                return key;
            }
            public V getValue(){
                return value;
            }
            public void setValue(V value){
                this.value = value;
            }
        }
        List<List<Node<K,V>>> bucket;
        public MyHashMap() {
            bucket = new ArrayList<>();
            for (int i = 0; i < size; i++){
                bucket.add(new LinkedList<>());
            }
        }

        public void put(K key, V value) {
            int idx = bucketHash(key);
            for (int i = 0; i < bucket.get(idx).size(); i++){
                if (bucket.get(idx).get(i).getKey() == key){
                    bucket.get(idx).get(i).setValue(value);
                    return;
                }
            }
            bucket.get(idx).add(new Node(key, value));
        }

        public V get(K key) {
            int idx = bucketHash(key);
            for (int i = 0; i < bucket.get(idx).size(); i++){
                if (bucket.get(idx).get(i).getKey() == key){
                    return bucket.get(idx).get(i).getValue();
                }
            }
            return null;
        }

        public void remove(K key) {
            int idx = bucketHash(key);
            for (int i = 0; i < bucket.get(idx).size(); i++){
                if (bucket.get(idx).get(i).getKey() == key){
                    bucket.get(idx).remove(i);
                    return;
                }
            }
        }

        private int bucketHash(K key){
            return key.hashCode() % size;
        }
    }
    public int minCostII(int[][] costs) {
        int n = costs.length;
        int k = costs[0].length;
        int[] dp = new int[k];
        for (int j = 0; j < k; j++){
            dp[j] = costs[0][j];
        }
        for (int i = 1; i < n; i++){
            int[] dpNew = new int[k];
            for (int c = 0; c < k; c++){
                int min_ = Integer.MAX_VALUE;
                for(int j = 0; j < k; j++){
                    if (c != j) min_ = Math.min(min_, costs[i-1][j]);
                }
                dpNew[c] = min_ + costs[i][c];
            }
            dp = dpNew;
        }
        return Arrays.stream(dp).min().getAsInt();
    }

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap obj = new MyHashMap();
 * obj.put(key,value);
 * int param_2 = obj.get(key);
 * obj.remove(key);
 */

}

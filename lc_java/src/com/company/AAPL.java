package com.company;

import java.io.*;
import java.util.*;
public class AAPL {
//    public static void main(String[] args) {
//        ArrayList<String> strings = new ArrayList<String>();
//        strings.add("Hello, World!");
//        strings.add("Welcome to CoderPad.");
//        strings.add("This pad is running Java " + Runtime.version().feature());
//
//        for (String string : strings) {
//            System.out.println(string);
//        }
//    }
    class OrderedStream{
        int ptr;
        String[] res;
        public OrderedStream(int n){
            ptr = 0;
            res = new String[n];
        }
        public List<String> insert(int id, String value){
            List<String> retval = new ArrayList<>();
            res[id-1] = value;
            while (ptr < res.length && res[ptr] != null){
                retval.add(res[ptr]);
                ptr++;
            }
            return retval;
        }
    }
    class LogSystem{
        List<String[]> timestamps = new LinkedList<>();
        List<String> units = Arrays.asList("Year", "Month", "Day", "Hour", "Minute", "Second");
        int[] indices = new int[]{4,7,10,13,16,19};
        public void put(int id, String timestamp){
            timestamps.add(new String[]{String.valueOf(id), timestamp});
        }
        public List<Integer> retrieve(String s, String e, String gra) {
            List<Integer> res = new LinkedList<>();
            int idx = indices[units.indexOf(gra)];
            for (String[] timestamp: timestamps){
                if (timestamp[1].substring(0, idx).compareTo(s.substring(0, idx)) >=0 &&
                timestamp[1].substring(0, idx).compareTo(e.substring(0, idx)) <= 0){
                    res.add(Integer.parseInt(timestamp[0]));
                }
            }
            return res;
        }
    }
    class LRUCache {
        class Node {
            int key;
            int value;
            Node prev;
            Node next;
            public Node(int key, int val){
                this.key = key;
                this.value = val;
            }
        }
        Map<Integer, Node> map;
        int capacity, count;
        Node head, tail;

        public LRUCache(int capacity) {
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.next = tail;
            tail.prev = head;
            this.capacity = capacity;
            count = 0;
            map = new HashMap<>();
        }
        public void moveToHead(Node node){
            node.next = head.next;
            head.next.prev = node;
            node.prev = head;
            head.next = node;
        }

        public void deleteNode(Node node){
            node.prev.next = node.next;
            node.next.prev = node.prev;
        }

        public int get(int key) {
            if (map.containsKey(key)){
                Node node = map.get(key);
                int result = node.value;
                deleteNode(node);
                moveToHead(node);
                return result;
            }
            return -1;
        }

        public void put(int key, int value) {
            if (map.containsKey(key)){
                Node node = map.get(key);
                node.value = value;
                deleteNode(node);
                moveToHead(node);
            }else{
                Node node = new Node(key, value);
                map.put(key, node);
                moveToHead(node);
                if (count == capacity){
                    map.remove(tail.prev.key);
                    deleteNode(tail.prev);
                }else {
                    count++;
                }
            }
        }
    }


/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
    class RandomizedSet {
        Map<Integer, Integer> val2index;
        List<Integer> nums;
        Random rand = new Random();
        public RandomizedSet() {
            val2index = new HashMap<>();
            nums = new ArrayList<>();
        }

        public boolean insert(int val) {
            if (! val2index.containsKey(val)){
                val2index.put(val, nums.size());
                nums.add(val);
                return true;
            }
            return false;
        }

        public boolean remove(int val) {
            if (!val2index.containsKey(val)) return false;
            int idx = val2index.get(val);
            if (idx == nums.size() - 1){
                val2index.remove(nums.size() - 1);
                return true;
            }
            int lastElem = nums.get(nums.size() - 1);
            nums.set(idx, lastElem);
            nums.remove(nums.size() - 1);
            val2index.remove(val);
            val2index.remove(lastElem);
            val2index.put(lastElem, idx);
            return true;
        }

        public int getRandom() {
            return nums.get(rand.nextInt(nums.size()));
        }
    }

// TreeMap:
// 	floorEntry(K key) less than or equal to the given key
//  lowerEntry(K key) strictly less than the given key,

//  ceilingEntry(K key) greater than or equal to the given key
//  higherEntry(K key) strictly greater than the given key
    class SnapshotArray {
        List<TreeMap<Integer, Integer>> arr;
        int snapId = 0;
        public SnapshotArray(int length) {
            arr = new ArrayList();

            for (int i = 0; i < length; i++) {
                arr.add(i, new TreeMap<>());
                arr.get(i).put(0, 0);
            }
        }

        public void set(int index, int val) {
            arr.get(index).put(snapId, val);
        }

        public int snap() {
            return snapId++;
        }

        public int get(int index, int snap_id) {
            return arr.get(index).floorEntry(snap_id).getValue();
        }
    }

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray obj = new SnapshotArray(length);
 * obj.set(index,val);
 * int param_2 = obj.snap();
 * int param_3 = obj.get(index,snap_id);
 */

    public class InsertOrReplaceArray{
        Map<Integer, Integer> nums;  // key: index, value: element
        Map<Integer, Integer> minIndex;  // key: element, value: minIndex
    public InsertOrReplaceArray(){
        nums = new HashMap<>();
        minIndex = new HashMap<>();
    }
    public void insertOrReplace(int elem, int index){
        nums.put(index, elem);
        if (minIndex.containsKey(elem)){
            minIndex.put(elem, Math.min(index, minIndex.get(elem)));
        }else{
            minIndex.put(elem, index);
        }
    }
    public int getMinIndex(int elem){
        if (! minIndex.containsKey(elem)) return -1;
        return minIndex.get(elem);
    }
}
    class Solution {
        public int longestConsecutive(int[] nums) {
            Set<Integer> set = new HashSet<>();
            for (int num : nums){
                set.add(num);
            }
            int max = 0;
            for (int num: nums){
                int tmp = num;
                if (!set.contains(tmp-1)){
                    while (set.contains(tmp)){
                        tmp++;
                    }
                    max = Math.max(max, num - tmp);
                }
            }
            return max;
        }
    }

//    Definition for a binary tree node.
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }

}

    public class Codec {

        // Encodes a tree to a single string.

        public String serialize(TreeNode root) {
            StringBuilder sb = new StringBuilder();
            buildString(root, sb);
            return sb.toString();
        }
        private void buildString(TreeNode root, StringBuilder sb){
            if (root == null) sb.append("#");
            else{
                sb.append(root.val).append(",");
                buildString(root.left, sb);
                buildString(root.right, sb);
            }
        }

        // Decodes your encoded data to tree.
        public TreeNode helper(Deque<String> nodes){
            String val = nodes.pollFirst();
            
            if (val.equals("#")){
                return null;
            }
            TreeNode res = new TreeNode(Integer.parseInt(val));
            res.left = helper(nodes);
            res.right = helper(nodes);
            return res;
        }
        public TreeNode deserialize(String data) {
            Deque<String> nodes = new ArrayDeque<>();
            nodes.addAll(Arrays.asList(data.split(",")));
            System.out.print(nodes);
            return helper(nodes);
        }
    }
    class Solution2 {
        public int compress(char[] chars) {
            int runner = 0, walker = 0;
            while (runner < chars.length){
                char currentChar = chars[runner];
                chars[walker++] = chars[runner];
                int count = 0;
                while (runner < chars.length && chars[runner] == currentChar){
                    count++;
                    runner++;
                }
                if (count == 1) continue;
                for (char c : Integer.toString(count).toCharArray()){
                    chars[walker++] = c;
                }
            }
            return walker;
        }
    }
    class MinStack <T extends  Comparable<T>>{
        Stack<T> mainStack = new Stack<>();
        Stack<T> minStack = new Stack<>();
        public void push(T x){
            mainStack.push(x);
            if (minStack.empty()){
                minStack.push(x);
            } else if (minStack.peek().compareTo(x) <= 0){
                minStack.push(x);
            }
        }

    }
    public class Car implements Comparable<Car> {
        private String model;
        private int topSpeed;

        // standard constructors, getters and setters
        public Car(String model, int speed){
            this.model = model;
            this.topSpeed = speed;
        }

        @Override
        public int compareTo(Car o) {
            return topSpeed - o.topSpeed;
        }
    }
    Car porsche = new Car("Porsche 959", 319);
    Car ferrari = new Car("Ferrari 288 GTO", 303);
    Car bugatti = new Car("Bugatti Veyron 16.4 Super Sport", 415);
    Car mcLaren = new Car("McLaren F1", 355);
    Car[] fastCars = { porsche, ferrari, bugatti, mcLaren };

    Car maxBySpeed = Arrays.stream(fastCars)
            .max((a, b)-> a.topSpeed - b.topSpeed)
            .orElseThrow(NoSuchElementException::new);

//    assertEquals(bugatti, maxBySpeed);


//    ConcurrentHashMap and SynchronizedHashMap
//    ConcurrentHashMap
//at a time any number of threads can perform retrieval operation but for updating in the object, the thread must lock the particular segment in which the thread wants to operate. This type of locking mechanism is known as Segment locking or bucket locking.
//    SynchronizedHashMap
//    If we need to perform thread-safe operations on it then we must need to synchronize it explicitly.
//    public static void main(String[] args) {
//        HashMap<Integer, String> hmap = new HashMap<Integer, String>();
//        Map<Integer, String> map = Collections.synchronizedMap(hmap);
//        synchronized (map){;}
//    }
    public class ListNode {
     int val;
     ListNode next;
     ListNode() {}
     ListNode(int val) { this.val = val; }
     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }
    class Solution3 {
        public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
            ListNode dummy = new ListNode(-1);
            ListNode head = dummy;
            int carry = 0;
            while (l1 != null && l2 != null){
                int num = l1.val + l2.val + carry;
                carry = num / 10;
                dummy.next = new ListNode(num % 10);
                dummy = dummy.next;
                l1 = l1.next;
                l2 = l2.next;
            }
            ListNode padding = l1 == null ? l2 : l1;
            while (padding != null){
                int num  = padding.val + carry;
                dummy.next = new ListNode(num % 10);
                carry = num / 10;
                dummy = dummy.next;
                padding = padding.next;
            }
            if (carry == 1) dummy.next = new ListNode(carry);
            return head.next;
        }
    }
    class SparseVector {

        HashMap<Integer, Integer> index2val;
        SparseVector(int[] nums) {
            index2val = new HashMap<>();
            for (int i = 0; i < nums.length; i++){
                if (nums[i] != 0) index2val.put(i, nums[i]);
            }
        }

        // Return the dotProduct of two sparse vectors
        public int dotProduct(SparseVector vec) {
            int res = 0;
            for (int index : this.index2val.keySet()){
                if (vec.index2val.containsKey(index)) res += vec.index2val.get(index) * this.index2val.get(index);
            }
            return res;
        }
    }
//    class UndergroundSystem {
////        private Map<String, Pair<Double, Double>> trips;
////        private Map<Integer, Pair<String, Integer>> checkins;
////        public UndergroundSystem() {
////            trips = new HashMap<>();
////            checkins = new HashMap<>();
//        }
//
//        public void checkIn(int id, String stationName, int t) {
//
//        }
//
//        public void checkOut(int id, String stationName, int t) {
//
//        }
//
//        public double getAverageTime(String startStation, String endStation) {
//
//        }
//    }

}


/*
 * To execute Java, please define "static void main" on a class
 * named Solution.
 *
 * If you need more classes, simply define them inline.
 */
/**
 Consider a simplified JSON which does not support Arrays, numbers or booleans.
 For Example:
 {
 "firstName": "John",
 "lastName": "Smith",
 "address": {
    "streetAddress": "21 2nd Street",
    "city": "New York",
 }
 }
 ---
 {
 "lastName": "Smith",
 "firstName": "John",
 "address": {
    "streetAddress": "21 2nd Street",
    "city": "New York",
 }
 }
 */
//enum Type {
// START_BRACE, END_BRACE, COLON, STRING
//}
//interface Token {
//    Type type;
//    String data;
//    String getData();
//    Type getType();
//}
//interface JsonStream {
//    Token getNextToken();
//}
//class JsonNode{
// HashMap<String, JsonNode> map;
// boolean nested;
// String value;
// JsonNode(String value){
// nested = false;
// map = null;
// this.value = value‍‌‌‌‍‍‌‌‌‌‌‌‍‍‌‍‍‍‍‌;
// }
// JsonNode(HashMap<String, JsonNode> map) {
// this.map = map;
// this.nested = true;
// }
//}
//class JsonStreamComparator implements Comparator<JsonStream> {
// boolean equals(JsonStream o1, JsonStream o2) {
// // Implement this.
//     return
// }
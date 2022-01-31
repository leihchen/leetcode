package com.company;
//
//import java.util.HashMap;
//import java.util.Map;
//
//class Solution1 {
//    class TreeNode {
//        char value;
//        TreeNode left;
//        TreeNode right;
//
//        TreeNode(char inp) {
//            this.value = inp;
//        }
//    }
//
//    public String sExp(char[][] pair) {
//        // build an adj matrix row is the parent node, col is the child node
//        int len = pair.length;
//        if (len == 0) return "";
//        int[][] adjMat = new int[26][26];
//        int[] inNode = new int[26];        // count the incoming edges of each node
//        int[] children = new int[26]; // count the number of children of each node
//        Map<Character, TreeNode> nodes = new HashMap();
//        TreeNode parent, child;
//        UnionFind myUnion = new UnionFind(26);
//        // build binary tree based on pair
//        for (int i = 0; i < len; i++) {
//            int row = pair[i][0] - 'A';
//            int col = pair[i][1] - 'A';
//            if (children[row] == 2 && adjMat[row][col] != 1) {
//                // more than 2 children but not the same edge
//                return "E1";
//            } else if (adjMat[row][col] == 1) {
//                // duplicated edges
//                return "E2";
//            } else if (myUnion.connected(row, col)) {
//                // if new link connect two that are already in same union there is loop
//                return "E3";
//            }
//            adjMat[row][col] = 1;
//            children[row]++;
//            inNode[col]++;
//            myUnion.connect(row, col);
//            connectNodes(pair, nodes, i);
//        }
//
//        // check multiple roots
//        int rNum = 0;
//        TreeNode root = null;
//        for (char x : nodes.keySet()) {
//            if (inNode[x - 'A'] == 0) {
//                rNum++;
//                if (rNum > 1) return "E4";
//                root = nodes.get(x);
//            }
//        }
//        if (root == null) return "E5";
//
//        // convert it to s-expression
//        return toSExpression(root);
//    }
//
//    // convert it to s-expression
//    String toSExpression(TreeNode root) {
//        if (root == null) return "";
//        return "(" + root.value + toSExpression(root.left) + toSExpression(root.right) + ")";
//    }
//
//    // connect parent and child node
//    private void connectNodes(char[][] pair, Map<Character, TreeNode> nodes,
//                              int i) {
//        TreeNode parent;
//        TreeNode child;
//        if (nodes.containsKey(pair[i][0])) parent = nodes.get(pair[i][0]);
//        else {
//            parent = new TreeNode(pair[i][0]);
//            nodes.put(pair[i][0], parent);
//        }
//        if (nodes.containsKey(pair[i][1])) child = nodes.get(pair[i][1]);
//        else {
//            child = new TreeNode(pair[i][1]);
//            nodes.put(pair[i][1], child);
//        }
//        if (parent.left == null) parent.left = child;
//        else {
//            if (parent.left.value < pair[i][1]) {
//                parent.right = child;
//            } else {
//                parent.right = parent.left;
//                parent.left = child;
//            }
//        }
//    }
//
//    public static void main(String[] args) {
//        Solution1 sln = new Solution1();
//        // (A(B(D(E(G))))(C(F)))
//        // char[][] pair = {{'B', 'D'}, {'D', 'E'}, {'A', 'B'}, {'C', 'F'}, {'E', 'G'}, {'A', 'C'}};
//        // E3
//        // char[][] pair = {{'A', 'B'}, {'A', 'C'}, {'B', 'D'}, {'D', 'C'}};
//        // (A(B(D)(G))(C(E(F))(H)))
//        // char[][] pair = {{'A', 'B'}, {'A', 'C'}, {'B', 'G'}, {'C', 'H'}, {'E', 'F'}, {'B', 'D'}, {'C', 'E'}};
//        // E1
//        // char[][] pair = {{'A', 'B'}, {'A', 'C'}, {'B', 'D'}, {'A', 'E'}};
//        // E2
//        // char[][] pair = {{'A', 'B'}, {'A', 'C'}, {'B', 'D'}, {'A', 'C'}};
//        // E4
//        char[][] pair = {{'A', 'B'}, {'A', 'C'}, {'B', 'G'}, {'C', 'H'}, {'E', 'F'}, {'B', 'D'}};
//        System.out.println(sln.sExp(pair));
//    }
//
//    public class UnionFind {
//        int[] pId;
//        int capacity;
//
//        public UnionFind(int x) {
//            pId = new int[x];
//            capacity = x;
//            for (int i = 0; i < capacity; i++) pId[i] = i;
//        }
//
//        public int parent(int i) {
//            int id = i;
//            while (pId[id] != id) {
//                id = pId[id];
//            }
//            int par = id;
//            // short path
//            while (pId[id] != id) {
//                int tmp = pId[id];
//                pId[id] = par;
//                id = tmp;
//            }
//            return par;
//        }
//
//        public boolean connected(int i, int j) {
//            return parent(i) == parent(j);
//        }
//
//        public void connect(int i, int j) {
//            int iid = parent(i), jid = parent(j);
//            if (iid != jid) {
//                pId[jid] = iid;
//            }
//        }
//    }
//
//
//    int minMovesToEvenFollowedByOdd(vector<int> arr) {
//        int res = 0, left = 0, right = arr.size() - 1;
//        // two-pointer approach
//        while(left < right) {
//            if(arr[left] % 2 != 0) {
//                while(right > left && arr[right] % 2 != 0) {
//                    // Find the first occurrence on the righthand side that can be swapped
//                    right--;
//                }
//
//                if(right > left) {
//                    // if we're already in the midpoint and the left pointer is odd then there is no swap
//                    res++;
//                    // handled this rightmost occurrence that was even => adjust to account for this
//                    right--;
//                }
//            }
//
//            left++;
//        }
//
//        return res;
//    }
//}


//public class Main {
//    public static boolean isIsomorphic(String s, String t) {
//        if (s.length() != t.length()) return false;
//        HashMap<Character, Character> mapping = new HashMap<Character, Character>();
//        for (int i = 0; i < s.length(); i++) {
//            if (mapping.containsKey(s.charAt(i)) && !mapping.get(s.charAt(i)).equals(t.charAt(i))) return false;
//            mapping.put(s.charAt(i), t.charAt(i));
//        }
//        return true;
//    }
//    public static int countBinarySubstrings(String s) {
//        int zeros = 0;
//        int ones = 0;
//        int res = 0;
//        char prev = s.charAt(0);
//        for (int i = 0; i < s.length(); ++i){
//            if (prev != s.charAt(i)) {
//                res += Math.min(zeros, ones);
//                prev = s.charAt(i);
//                if (s.charAt(i) == '1') ones = 0;
//                else zeros = 0;
//            }
//            if (s.charAt(i) == '1') ones++;
//            else zeros++;
//        }
//        return res + Math.min(zeros, ones);
//    }
//    public static String cleanSpace(char[] s){
//        int i = 0, j = 0;
//        int n = s.length;
//        while(j < n){
//            while (j < n && s[j] == ' ') j++;
//            while (j < n && s[j] != ' ') s[i++] = s[j++];
//            while (j < n && s[j] == ' ') j++;
//            if (j < n) s[i++] = ' ';
//        }
//        return new String(s).substring(0, i);
//    }
//    public static void main(String[] args) {
//	// write your code here
//        int res = countBinarySubstrings("00011100");
//        System.out.println(res); // prints Hello World
//    }
//}


//import java.time.Instant;
//import java.util.ArrayList;
//import java.util.HashMap;
//import java.util.Map;
//
//class AkunaHashTableFromEventStream {
//    /**
//     * A class which constructs a view of the Hash Table's state given the input
//     * DML Events.
//     */
//    public static class HashTable {
//        private Map<String, String> table;
//        private long highWatermark = -1;
//
//        public HashTable(ArrayList<String> rawEvents) {
//            table = new HashMap<>();
//            for (String rawEvent : rawEvents) {
//                String[] eInfo = rawEvent.split("\\|");
//                String type = eInfo[1];
//                String key = eInfo[2];
//                highWatermark = Long.parseLong(eInfo[0]);
//                switch (type) {
//                    case "INSERT":
//                        if (!table.containsKey(key)) {
//                            table.put(key, eInfo[3]);
//                        }
//                        break;
//                    case "UPSERT":
//                        table.put(key, eInfo[3]);
//                        break;
//                    case "DELETE":
//                        table.remove(key);
//                        break;
//                }
//            }
//        }
//
//        /**
//         * Retrieve the state of the HashTable after applying all input events
//         *
//         * @return a Map representing the Keys and Values of the current state
//         */
//        public Map<String, String> getTable() {
//            return table;
//        }
//
//        /**
//         * Retrieve the high-watermark of the HashTable as the millisecond epoch
//         * timestamp of the latest event read or Instant.EPOCH in the case where
//         * no event occurred.
//         *
//         * @return an Instant representing the high watermark
//         */
//        public Instant getHighWatermark() {
//            return highWatermark == -1 ? Instant.EPOCH : Instant.ofEpochMilli(highWatermark);
//        }
//    }
//}


// Feel free to add any imports you need

//class Solution {
//    /**
//     * A class which wraps a raw binary WAL and reconstructs DML Events.
//     */
//    public static class WAL {
//        /**
//         * Construct the WAL
//         * @param input the raw binary WAL
//         */
//        private ArrayList eventsList;
//        private Map<Integer, String> IDToName = new HashMap<Integer, String>(){{
//            put(0, "INSERT");
//            put(1, "UPSERT");
//            put(2, "DELETE");
//        }};
//        // private Map<String, String> IDToName = new HashMap<String, String>(){{
//        //     put("0", "INSERT");
//        //     put("1", "UPSERT");
//        //     put("2", "DELETE");
//        // }};
//
//        public WAL(byte[] input) {
//            //
//            eventsList = new ArrayList();
//            int i = 0;
//            final int BYTE_ARRAY_LENGTH = input.length;
//            while (i < BYTE_ARRAY_LENGTH) {
//                StringBuilder sb = new StringBuilder();
//
//                Long epochMilli = ByteBuffer.wrap(input, i, 8).getLong();
//                String epochMilliString = String.valueOf(epochMilli);
//                sb.append(epochMilliString).append("|");
//                i += 8;
//
//                int id = input[i] & 0xFF;
//                String name = IDToName.get(id);
//                sb.append(name).append("|");
//                i += 1;
//
//                Short keyLength = ByteBuffer.wrap(input, i, 2).getShort();
//                // String keyLengthString = String.valueOf(keyLength);
//                // System.out.println("keyLengthString" + keyLengthString);
//                i += 2;
//
//
//                //String asciiString = Arrays.toString(ascii);
//                Charset characterSet = StandardCharsets.US_ASCII;
//                String key = new String(input, i, keyLength, characterSet);
//                //String key = Arrays.toString(keyByteArray);
//                sb.append(key);
//                i += keyLength;
//
//                if (name.equals("DELETE")) {
//                    //Do nothing
//                } else {
//                    Short valueLength = ByteBuffer.wrap(input, i, 2).getShort();
//                    //String valueLengthString = String.valueOf(valueLength);
//                    i += 2;
//
//                    String value = new String(input, i, valueLength, characterSet);
//                    //byte[] valueByteArray = ByteBuffer.wrap(input, i, valueLength).array();
//                    //String value = new String(valueByteArray, characterSet);
//                    //String value = Arrays.toString(valueByteArray);
//                    sb.append("|").append(value);
//                    i += valueLength;
//                }
//
//                eventsList.add(sb.toString());
//            }
//        }
//
//        /**
//         * Retrieve all events contained within the WAL as their string values in time order
//         * DML Event String Format: "<epoch_milli>|<message_name>|<key>|<value>"
//         *
//         * @return a time-ordered sequence of DML Event strings
//         */
//        public ArrayList<String> getEvents() {
//            //
//            return eventsList;
//        }
//    }
//}
public class tiktokAkuna {
}

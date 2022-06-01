package com.company;//package com.company;
import java.awt.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.Inet4Address;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.Consumer;
//import org.apache.commons.math4.util.MathUtils;


// LC 239 https://leetcode.com/problems/sliding-window-maximum/
// Counting Pairs, LC 532
// PalindromeSubseq product of two parts
// max profit 不一样的地方只是上一个的结束时间和下一个的开始时间上不能挨着，i.e. [1, 3], [3, 4]因为3是挨着的，只能选其中一个
// GridLand, kth largest path with 'HV'
// maxSquareSubGrind size with sum smaller than k
// reconstructing array
// closest-pair-of-points https://www.geeksforgeeks.org/closest-pair-of-points-onlogn-implementation/
// profit target
// substring calculator
// Maximum of minimum difference of all pairs from subsequences of given size https://www.geeksforgeeks.org/maximum-of-minimum-difference-of-all-pairs-from-subsequences-of-given-size/
// global max

// * path to goal by moving left and right
// * some other https://www.1point3acres.com/bbs/thread-678726-1-1.html
// Algorithm L
class Solution {
    // 用一个二维matrix记录， dp[j] 代表这一串字符index i和j之间最大的palindrome subsequence的长度， 有了这个dp table之后就算出来 dp[0][n]*dp[n+1][m]的最大值， 其中 0 <= n <= m, m就是输入的字符串长度 - 1
    public static int longestPalindromeSubseq(String s) {
        int n = s.length();
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; i++){
            dp[i][i] = 1;
        }
        // dp[i][j] = dp[i+1][j-1] + 2 if dp[i] == dp[j]
        // otherwise = max(dp[i][j], dp[i+1][j], dp[i][j-1])
        for (int i = n-1; i >= 0; i--){
            for (int j = i+1; j < n; j++){
                if (s.charAt(i) == s.charAt(j)){
                    dp[i][j] = 2 + dp[i+1][j-1];
                }
                dp[i][j] = Math.max(dp[i][j], Math.max(dp[i+1][j], dp[i][j-1]));
            }
        }
        int res = 0;
        for (int i = 0; i < n-1; i ++){
            res = Math.max(res, dp[0][i] * dp[i+1][n-1]);
        }
        return res;
    }

    // https://leetcode.com/discuss/interview-question/1028649/snowflake-oa-maximum-order-volume&#8205;&#8204;&#8204;&#8204;&#8205;&#8205;&#8204;&#8204;&#8204;&#8204;&#8204;&#8204;&#8205;&#8205;&#8204;&#8205;&#8205;&#8205;&#8205;&#8204;
    // https://leetcode-cn.com/problems/maximum-profit-in-job-scheduling/
    public static int jobScheduling(int[] startTime, int[] duration, int[] profit) {
        int n = startTime.length;
        int[][] jobs = new int[n][3];
        for (int i = 0; i < n; i++) {
            jobs[i] = new int[] {startTime[i], startTime[i] + duration[i], profit[i]};
        }
        Arrays.sort(jobs, (a, b)->a[1] - b[1]);
        TreeMap<Integer, Integer> dp = new TreeMap<>();
        dp.put(0, 0);
        for (int[] job : jobs) {
            int cur = dp.lowerEntry(job[0]).getValue() + job[2]; // lowerEntry strictly less than, floorEntry less than
            if (cur > dp.lastEntry().getValue())
                dp.put(job[1], cur);
        }
        return dp.lastEntry().getValue();
    }
    static int factorial(int n){
        if (n == 0)
            return 1;
        else
            return(n * factorial(n-1));
    }
    public static String getKthPermutation(int n, int k){
        List<Integer> num = new LinkedList<Integer>();
        for (int i = 1; i <= n; i++){
            num.add(i);
        }
        int[] fact = new int[n];
        fact[0] = 1;
        for (int i = 1; i < n; i++){
            fact[i] = fact[i-1] * i;
        }
        k--;
        StringBuilder sb = new StringBuilder();
        for (int i = n-1; i >= 0; i--){
            int idx = k / fact[i];
            k = k % fact[i];
            sb.append(num.get(idx));
            num.remove(idx);
        }
        return sb.toString();
    }

    // redo gridLand: use x <= 10 and y <= 10
//    https://leetcode.com/discuss/interview-question/527769/lucid-oa-gridland
    public static List<String> getSafePaths(String[] journeys){
        int[][] dp = new int[11][11];
        for (int[] row : dp){
            Arrays.fill(row, 1);
        }
        for (int i = 1; i < 11; i++){
            for (int j = 1; j < 11; j++){
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
            }
        }
        List<String> res = new ArrayList<>();
        for (String j : journeys){
            String[] xyk = j.split(" ");
            int x = Integer.parseInt(xyk[0]), y = Integer.parseInt(xyk[1]), k = Integer.parseInt(xyk[2]) + 1;
            StringBuilder sb = new StringBuilder();
            while (x > 0 && y > 0){
                if (dp[x-1][y] >= k){
                    sb.append("H");
                    x -= 1;
                }else{
                    sb.append("V");
                    y -= 1;
                    k -= dp[x-1][y];
                }
            }
            for (int i = 0; i < x; i++) sb.append("H");
            for (int i = 0; i < y; i++) sb.append("V");
            res.add(sb.toString());
        }
        return res;
    }
    // https://www.hackerrank.com/challenges/lexicographic-steps/problem
    public static String getOnePath(int x, int y, int k){
        StringBuilder sb = new StringBuilder();
        while (x > 0 && y > 0){
            int base = factorial(x + y - 1) / factorial(x - 1) / factorial(y);
            if (k >= base){
                sb.append("V");
                y--;
                k -= base;
            }
            else{
                sb.append("H");
                x --;
            }
        }
        for (int i = 0; i < x; i++) sb.append("H");
        for (int i = 0; i < y; i++) sb.append("V");
        return sb.toString();
    }

    public static int maxSquareSubGrind(int[][] matrix, int k){
        int n = matrix.length;
        int dp[][] = new int[n+1][n+1];
        for (int i = 1; i <= n; i++){
            for (int j = 1; j <= n; j++){
                dp[i][j] = matrix[i-1][j-1] + dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1];
            }
        }
        int left = 0;
        int right = n;
        while (left <= right){
            int mid = (right + left) / 2;
            int res = 0;
            for (int x = 0; x <= n - mid; x++){
                for (int y = 0; y <= n - mid; y++){
                    res = Math.max(res, dp[x+mid][y+mid] - dp[x][y+mid] - dp[x+mid][y] + dp[x][y]);
                }
            }
            if (res == k){
                return mid;
            }else if(res > k){
                right = mid - 1;
            }else{
                left = mid + 1;
            }
        }
        return right;
    }

    // https://leetcode.com/discuss/interview-question/515762/Dunzo-or-OA-2020-or-Reconstructing-Arrays
    public static List<Integer> solve(List<Integer> n, List<Integer> m, List<Integer> totalCost)
    {
        int MAXN = 55;
        int MAXM = 105;
        int MAXCOST = 55;
        double MOD = 1e9 + 7;
        List<Integer> finalOutput = new ArrayList<Integer>();

        for(int ar = 0 ; ar < n.size() ; ar++) {
            int nValue = n.get(ar);
            int mValue = m.get(ar);
            int costValue = totalCost.get(ar);

            int cum[][][]=new int[MAXN][MAXM][MAXCOST];
            long dp[][][]=new long [MAXN][MAXM][MAXCOST];
            for (int j = 1; j <= m.get(ar); j++) {
                dp[1][j][0] = 1;
                cum[1][j][0] = j;
            }

            for (int i = 2; i <= nValue; i++) {
                for (int j = 1; j <= mValue; j++) {
                    for (int k = 0; k <= costValue; k++) {
                        dp[i][j][k] = (j * dp[i - 1][j][k]);  // <== construct dp[i,j,k] without changing max and cost
                        // if we don't change to cost and maxValue, just add any number at the back (j choice avail)
                        dp[i][j][k] %= MOD;
                        if(k!= 0)
                            dp[i][j][k] += cum[i - 1][j - 1][k-1];   // <== construct dp[i,j,k] by adding 1 to len, max, and cost
                        // from cum[i - 1][j - 1][k-1], add digit j at the end, increasing cost by 1
                        dp[i][j][k] %= MOD;
                        cum[i][j][k] = (int)((cum[i][j - 1][k] + dp[i][j][k])%MOD);
                    }
                }
            }
            finalOutput.add(cum[nValue][mValue][costValue]);
        }
        return finalOutput;

    }
    // https://leetcode.com/discuss/interview-question/881527/paths-to-a-goal-hackerrank
    public static int pathToGoal(){return 0;}

    // O(nlogn) find min dist of closest pair of points
    static double dist(Point a, Point b){
        return Math.sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    }

    static double stripClosest(Point[] strip, int n, double d){
        double min = d;
        for (int i = 0; i < n; i++){
            for (int j = i+1; j < n && (strip[j].y - strip[i].y) < min; j++){
                min = Math.min(d, dist(strip[i], strip[j]));
            }
        }
        return min;
    }
    static double closestUtil(Point[] Px, Point[] Py){
        int n = Px.length;
        if (n <= 3){
            double res = Double.POSITIVE_INFINITY;
            for (int i = 0; i < n; i++){
                for (int j = i+1; j < n; j++){
                    res = Math.min(res, dist(Px[i], Py[i]));
                }
            }
            return res;
        }
        int mid = n / 2;
        Point midPoint = Px[mid];
        Point[] Pyl = new Point[mid];
        Point[] Pyr = new Point[mid];
        int li = 0, ri = 0;
        for (int i = 0; i < n; i++){
            if ((Py[i].x < midPoint.x || (Py[i].x == midPoint.x && Py[i].y < midPoint.y) && li < mid)){
                Pyl[li++] = Py[i];
            }else{
                Pyr[ri++] = Py[i];
            }
        }
        double dl = closestUtil(Arrays.copyOfRange(Px, 0, mid), Pyl);
        double dr = closestUtil(Arrays.copyOfRange(Px, mid, n), Pyr);
        double d = Math.min(dl, dr);
        Point[] strip = new Point[n];
        int j = 0;
        for (int i = 0; i < n; i++){
            if (Math.abs(Py[i].x - midPoint.x) < d){
                strip[j++] = Py[i];
            }
        }
        return stripClosest(strip, j, d);
    }
    public static double closestPairDist(Point[] points){
        int n = points.length;
        Point[] Px = new Point[n];
        Point[] Py = new Point[n];
        for (int i = 0; i < n; i++){
            Px[i] = points[i];
            Py[i] = points[i];
        }
        Arrays.sort(Px, (a, b)->a.x - b.x);
        Arrays.sort(Py, (a, b)->a.y - b.y);
        return closestUtil(Px, Py);
    }
    public static int profitTargets(int[] profits, int target){
        Set<Integer> seen = new HashSet<>();
        Set<Integer> d = new HashSet<>();
        int res = 0;
        for (int x : profits){
            if (d.contains(target-x) && !seen.contains(x)){
                res ++;
                seen.add(x);
            }else if (!d.contains(target-x)){
                d.add(x);
            }
        }
        return res;

    }
    static int LCP(String s1, String s2){
        int n = Math.min(s1.length(), s2.length());
        int res = 0;
        for (int i = 0; i < n; i++){
            if (s1.charAt(i) != s2.charAt(i)) break;
            else res = i + 1;
        }
        return res;
    }
    public static int substringCal(String s){
        int n = s.length();
        String[] suffix = new String[n];
        for (int i = 0; i < n; i++){
            suffix[i] = s.substring(i);
        }
        int lcpSum = 0;
        Arrays.sort(suffix, (a, b) -> a.compareTo(b));
        for (int i = 0; i < n-1; i++){
            lcpSum += LCP(suffix[i], suffix[i+1]);
        }
        return n * (n+1) / 2 - lcpSum;
    }

    static boolean existSubseq(int[] arr, int tar, int m){
        // if possible to choose a subseq of arr s.t. min(pairwise diff)  >= tar
        int cnt = 1, prev = arr[0], n = arr.length;
        for (int i = 0; i < n; i++){
            if (cnt == m){
                return true;
            }
            if (arr[i] - prev >= tar){
                cnt ++;
                prev = arr[i];
            }
        }
        return false;
    }
    public static int globalMaximum(int[] arr, int m){
        int n = arr.length;
        int start = 0, end = arr[n-1] - arr[0];
        while (start <= end){
            int mid = (start + end) / 2;
            if (existSubseq(arr, m, mid)){
                start = mid + 1;
            }else{
                end = mid - 1;
            }
        }
    return start;
    }

    public static String domino(String[] A){
        Set<String> seen = new HashSet<>();
        for (String a : A){
            seen.add(a);
            seen.add(String.valueOf(a.charAt(2)) + '-' + a.charAt(0));
        }
        for (int i = 0; i <= 6; i++){
            for (int j = 0; j <= i; j++){
                String result1 = Integer.toString(i) + '-' + j;

                System.out.println(result1);
                if (!seen.contains(result1)){
                    return result1;
                }
            }
        }
        return "";
    }

    public static void main(String[] args){
//        BlockingDeque<Integer> q = new LinkedBlockingDeque<>();
//        Arrays.asList(3,2,1).forEach((d)->{
//            new Thread( () -> {
//                try {
//                    Thread.sleep(d * 1000);
//                } catch (InterruptedException e){
//                    e.printStackTrace();
//                }
//                q.add(d);
//            }).start();
//        });
//        System.out.println(q.take());
//        int[] l = {1};
//        System.out.println(Arrays.binarySearch(l, -1));
//        Syntax.MyHashMap<Integer, Integer> map = new Syntax.MyHashMap<>();
//        map.put(1,1);
//        System.out.println(map.get(1));
       System.out.println(domino(new String[] {"0-0", "1-1", "2-2", "3-3", "4-4", "5-5", "6-6", "0-1", "2-3"}));

    }
//    public static void main(String[] args) throws IOException {
//        int a = 0;
//        List<Integer> b = new ArrayList<>();
//        func(a, b);
//        System.out.println(a + b.toString());
//        Consumer<List<String>> delteBlankItems = (items) -> {
//            for (int i = 0; i < items.size(); i++ ){
//                if  (items.get(i).length() == 0){
//                    System.out.println(i);
//                    items.remove(i);
//                }
//            }
//        };
//        List<String> names = new ArrayList<>(Arrays.asList("aa", "bb", "", "", "cv"));
//        delteBlankItems.accept(names);
//
//        System.out.println(names);
//        byte[] data = null;
//        FileInputStream stream = null;
//        File file = new File("example.txt");
//        try {
//            stream = new FileInputStream(file);
//            data = stream.readAllBytes();
//        } catch (java.io.IOException ignored){
//
//        } finally {
//            stream.close();
//        }
//    }

//    public static void main(String[] args) throws InterruptedException, ExecutionException, TimeoutException {
//        // Mutlithreading 1
////        Multithreading myThing = new Multithreading();
////        myThing.start();
////        synchronized (myThing) {
////            myThing.wait();
////        }
////        System.out.println(myThing.sum);
//
//        // Mutlithreading 2
//        // callable allows retval
//        class TaskRetval implements Callable<String> {
//            final int base;
//            TaskRetval(int n) {base = n;}
//            @Override
//            public String call() throws Exception{
//                StringBuilder sb = new StringBuilder();
//                for (int i = 0; i < 10; i++){
//                    sb.append((char)(i + base + 'a'));
//                    System.out.println((char)(i + base + 'a') + Integer.toString(base));
//                    Thread.sleep(500);
//                }
//                return sb.toString();
//            }
//        }
//        // runnable don't
////        class Task implements Runnable{
////            public void run(){
////                System.out.println("Thred Name" + Thread.currentThread().getName());
////            }
////        }
////
//        ScheduledExecutorService service = Executors.newScheduledThreadPool(10);
//        Future<String> future = service.submit(new TaskRetval(0));  // submit a callable, make a placeholder to retval
//        Future<String> future2 = service.submit(new TaskRetval(10));
//        String res = future.get() + '+' + future2.get();
//        System.out.println(res);
//        service.shutdown();
////        service.execute(new Task());  // execute a runnable
////        future.cancel(false);  // remove task from blocking queue if not yet scheduled, otherwise no effect
////        Integer result = future.get(1, TimeUnit.SECONDS);  // get is blocking operation with timeout of 1 second
////
////        ExecutorService service1 = new ThreadPoolExecutor(10, 100, 120, TimeUnit.SECONDS, new ArrayBlockingQueue<>(300));
////
////        // CompletableFuture
////        ExecutorService cpuBound = Executors.newFixedThreadPool(16);
////        ExecutorService ioBound = Executors.newCachedThreadPool();
////        for (int i = 0; i < 5; i++) {
////            CompletableFuture.supplyAsync(() -> task1(), cpuBound)  // unblock 5 for loops
////                    .thenApplyAsync(x -> task2(x), ioBound)
////                    .thenApply(x -> task3(x))
////                    .thenAccept(x -> taskFinal(x));
////            // can add Async to all methods, Async allows other threads to executor the task
////            // for example different type of tasks (io/cpu bounded), this
////        }
////
////        // Timeout a thread
////        myThing.stop();
////        try {
////            future.get(10, TimeUnit.SECONDS);
////        } catch (InterruptedException | ExecutionException e){
////
////        } catch (TimeoutException e){
////            future.cancel(true);
////        }
//
//        // Callable -> Future.cannel
//        // ExecutorService.shutdown/shutdownNow
//
//        // ForkJoinPool
//        class Fib extends RecursiveTask<Integer>{
//            final int n;
//            Fib(int n) {this.n = n;}
//
//            public Integer compute(){
//                if (n <= 2) {
//                    return n;
//                }
//                System.out.println("number=" + n);
//                Fib fcal1 = new Fib(n - 1);
//                fcal1.fork();
//                Fib fcal2 = new Fib(n - 2);
//                int fcal2Compute = fcal2.compute();
//                int fcal1Join = fcal1.join();
//                System.out.println("number=" + n + " fcal1.join()=" + fcal1Join + " fcal2.compute()=" + fcal2Compute);
//                return fcal2Compute + fcal1Join;
//            }
//        }
//
////        Fib myFib = new Fib(5);
////        Integer x = myFib.compute();
////        System.out.println(x);
//
//
////       int t1 = longestPalindromeSubseq("bbbab");
////        System.out.println(t1);
////        int t2 = longestPalindromeSubseq("attract");
////        System.out.println(t2);
//
////    int[] start = new int[] {10, 5, 15, 18, 30};
////    int[] duration  = new int[] {30, 12, 20, 35, 35};
////    int[] volume = new int[] {50, 51, 20, 25, 10};
////    int res1 = jobScheduling(start, duration, volume);
////    System.out.println(res1);
//
////    String prev = "";
////    for (int i = 0; i < 35; i++){
////            String res = getOnePath(3,4,i);
////            System.out.println(res + (prev.compareTo(res) < 0));
////            prev = res;
////        }
////    int[][] test = new int[][] {new int[] {1,1,1,1}, new int[] {2,2,2,2}, new int[] {3,3,3,3}, new int[] {4,4,4,4}};
////        System.out.println(getSafePaths(new String[] {"2 2 2", "2 2 3", "2 2 0", "2 2 5"}));
////    System.out.println(maxSquareSubGrind(test, 20));
////        Point[] P = {new Point(2, 3), new Point(12, 30),
////                new Point(40, 50), new Point(5, 1),
////                new Point(12, 10), new Point(3, 4)};
//
////        System.out.println(closestPairDist(P));
//
////        System.out.println(profitTargets(new int[] {5,7,9,13,11,6,6,3,3}, 12));
////        System.out.println("kincenvizh".substring(9));
////        System.out.println(substringCal("kincenvizh"));
////        System.out.println(globalMaximum(new int[] {1,2,3,4}, 3));
//
//    }
    public List<String> topKFrequent(String[] words, int k) {
        List<String> res = new LinkedList<>();
        HashMap<String, Integer> map = new HashMap<>();
        for (String word: words){
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        PriorityQueue<Map.Entry<String, Integer>> pq = new PriorityQueue<>((a, b)-> a.getValue() == b.getValue() ? b.getKey().compareTo(a.getKey()) : a.getValue() - b.getValue());
        for (Map.Entry<String, Integer> entry : map.entrySet()){
            pq.offer(entry);
            if (pq.size() > k)
                pq.poll();

        }
        while(!pq.isEmpty())
            res.add(0, pq.poll().getKey());
        return res;
    }
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) return res;
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);
        while (!q.isEmpty()){
            int cnt = q.size();
            for (int i = 0; i < cnt; i++){
                TreeNode cur = q.poll();
                if (i == cnt - 1) res.add(cur.val);
                if (cur.left != null) q.offer(cur.left);
                if (cur.right != null) q.offer(cur.right);
            }
        }
        return res;
    }
}

//import java.io.*;
//        import java.math.*;
//        import java.security.*;
//        import java.text.*;
//        import java.util.*;
//        import java.util.concurrent.*;
//        import java.util.function.*;
//        import java.util.regex.*;
//        import java.util.stream.*;
//        import static java.util.stream.Collectors.joining;
//        import static java.util.stream.Collectors.toList;
//
//
//
//class Result {
//
//    /*
//     * Complete the 'substringCalculator' function below.
//     *
//     * The function is expected to return a LONG_INTEGER.
//     * The function accepts STRING s as parameter.
//     */
//
//    static int lcp(String s1, String s2){
//        int n = Math.min(s1.length(), s2.length());
//        int res = 0;
//        for (int i = 0; i < n; i++){
//            if (s1.charAt(i) != s2.charAt(i)) break;
//            else res = i + 1;
//        }
//        return res;
//    }
//
//    public static long substringCalculator(String s) {
//        // Write your code here
//        int n = s.length();
//        String[] suffix = new String[n];
//        for (int i = 0; i < n; i++){
//            suffix[i] = s.substring(i);
//        }
//        long lcpSum = 0;
//        Arrays.sort(suffix, (a,b) -> a.compareTo(b));
//        for (int i = 0; i < n-1; i++){
//            lcpSum += lcp(suffix[i], suffix[i+1]);
//        }
//        return n * (n + 1) / 2 - lcpSum;
//    }
//    // Set<String> seen = new HashSet<>();
//    // for (int i = 0; i <= s.length(); i++){
//    //     for (int j = i + 1; j <= s.length(); j++){
//    //         seen.add(s.substring(i, j));
//    //     }
//    // }
//    // return seen.size();
//    // }
//}
//
//public class Solution {
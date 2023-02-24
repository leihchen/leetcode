import java.math.BigInteger;
import java.util.*;

public class snowflake {
    /*
    https://leetcode.com/discuss/interview-question/3111290/SnowFlake-or-OA-Questions
    https://www.1point3acres.com/bbs/thread-959453-1-1.html
    maximize array value ok
    palindrome subseq ok https://leetcode.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/discuss/1458289/Mask-DP
    task scheduler  ok https://leetcode.com/discuss/interview-question/2775415/SnowFlake-OA
    array reduction
    4
    cross the threshold ok
     */
    public int wordsKDistinctContVowels(int n, int k){
        int MOD = 1000000007;
        int[][] dp = new int[n + 1][k + 1];
        int sum = 1;
        for (int i = 1; i <= n; i++) {
            dp[i][0] = sum * 21;
            dp[i][0] %= MOD;
            sum = dp[i][0];
            for (int j = 1; j <= k; j++){
                if (j > i) dp[i][j] = 0;
                else if (j == i) dp[i][j] = BigInteger.valueOf(5).modPow(BigInteger.valueOf(i), BigInteger.valueOf(MOD)).intValue();
                else {
                    dp[i][j] = dp[i - 1][j - 1] * 5;
                }
                dp[i][j] %= MOD;
                sum += dp[i][j];
                sum %= MOD;
            }
        }
        return sum;
    }

    public static int maxSquareSubGrind(int[][] matrix, int k){
        int n = matrix.length;
        int[][] dp = new int[n+1][n+1];
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

    public int countVowelSubstrings(String word) {
        Set<Character> v = new HashSet<>(Arrays.asList('a', 'e', 'i', 'o', 'u'));
        int ans = 0;
        int prev = -1;

        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);

            if (!v.contains(c)) {
                if (prev == -1) {
                    prev = i;
                    continue;
                } else {
                    ans += allfive(word.substring(prev + 1, i));
                    prev = i;
                }
            }
        }

        ans += allfive(word.substring(prev + 1, word.length()));
        return ans;
    }

    private int allfive(String word) {
        int res = 0;
        int left = 0;
        int n = word.length();
        Map<Character, Integer> d = new HashMap<>();

        for (int i = 0; i < n; i++) {
            char c = word.charAt(i);
            d.put(c, d.getOrDefault(c, 0) + 1);

            while (d.get(word.charAt(left)) > 1) {
                d.put(word.charAt(left), d.get(word.charAt(left)) - 1);
                left++;
            }

            if (valid(d)) {
                res += left + 1;
            }
        }

        return res;
    }

    private boolean valid(Map<Character, Integer> d) {
        for (char c : Arrays.asList('a', 'e', 'i', 'o', 'u')) {
            if (d.getOrDefault(c, 0) < 1) {
                return false;
            }
        }

        return true;
    }

    public int minArrayValue(int[] inp) {
        int n = inp.length;
        int[] pre = new int[n];
        int[] dp = new int[n];
        pre[0] = inp[0];
        dp[0] = inp[0];

        for (int i = 1; i < n; i++) {
            pre[i] = pre[i - 1] + inp[i];
        }

        for (int i = 1; i < n; i++) {
            if (inp[i] <= dp[i - 1]) {
                dp[i] = dp[i - 1];
            } else {
                dp[i] = Math.max(dp[i - 1], (int) Math.ceil((double) pre[i] / (i + 1)));
            }
        }

        return dp[n - 1];
    }

    public static int getMaxBarrier(int[] initialEnergy, int th) {
        int maxEnergy = Arrays.stream(initialEnergy).max().getAsInt();
        int left = 0;
        int right = maxEnergy;

        while (left <= right) {
            int mid = (right - left) / 2 + left;
            int sum_m = getSum(initialEnergy, mid);

            if (sum_m == th) {
                return mid;
            } else if (sum_m > th) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        return right;
    }

    private static int getSum(int[] initialEnergy, int barrier) {
        int ans = 0;

        for (int i : initialEnergy) {
            if (i - barrier > 0) {
                ans += i - barrier;
            }
        }

        return ans;
    }

    public static void main(String[] args) {


        snowflake s = new snowflake();
        // test wordsKDistinctContVowels
//        System.out.println(s.wordsKDistinctContVowels(1, 1));
//        System.out.println(s.wordsKDistinctContVowels(4, 1));
//        System.out.println(s.wordsKDistinctContVowels(3, 3));

        // test maxSquareSubGrind
//        int[][] test = new int[][] {new int[] {1,1,1,1}, new int[] {2,2,2,2}, new int[] {3,3,3,3}, new int[] {4,4,4,4}};
//        int[][] test2 = new int[][] {new int[] {2,2,2}, new int[] {3,3,3}, new int[] {4,4,4},};
//        System.out.println(maxSquareSubGrind(test2, 4));
        // test countVowelSubstrings
//        System.out.println(s.countVowelSubstrings("aeioaexaaeuiou"));

        // test minArrayValue
//        System.out.println(s.minArrayValue(new int[] {10,3,5,7}));
        // test  Solution
        int c[] = {5,6,7,8,8,10};
        int t[] = {1,1,1,1,1,10};
        Solution sol = new Solution();
        System.out.println(sol.func(c, t));
    }
}

class Solution {
    int n;
    int[] c;
    int[] t;
    int[] sufsum;
    Map<String, Integer> memo;


    public int f(int i, int j) {
        if (j + (i < n ? sufsum[i] : 0) < 0) {
            return Integer.MAX_VALUE;
        }

        if (j >= n - i) {
            return 0;
        }

        String key = i + ":" + j;
        if (memo.containsKey(key)) {
            return memo.get(key);
        }

        int result = Math.min(c[i] + f(i + 1, j + t[i]), f(i + 1, j - 1));
        memo.put(key, result);
        return result;
    }

    public int func(int[] c, int[] t) {
        n = t.length;
        this.c = c;
        this.t = t;
        memo = new HashMap<>();
        this.sufsum = new int[n];
        for (int i = n - 1; i >= 0; i--) {
            sufsum[i] = t[i] + (i + 1 < n ? sufsum[i + 1] : 0);
        }
        System.out.println(Arrays.toString(sufsum));
        return f(0, 0);
    }
}

package test;//Maximum size of ribbon
//Given a list representing the length of ribbon, and the target number "k" parts of ribbon.
//we want to cut ribbon into k parts with the same size, at the same time we want the maximum size.
//Ex.
//Input: A = [1, 2, 3, 4, 9], k = 5 Output: 3

public class Ribbon {
    public int solution(int[] A, int k){
        int hi = 0;
        for(int i = 0; i < A.length; i++) {
            hi += A[i];
        }
        int lo = 0;
        int res = 0;
        while(lo <= hi) {
            int mid = (lo + hi) / 2;
            int part = 0;
            for(int i = 0; i < A.length; i++) {
                part += A[i]/mid;
            }
            if(part >= k) {
                res = Math.max(res, mid);
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return res;


    }



    public static void main(String[] args){
        int[] A = {1,2,3,4,9};
        int k=5;
        System.out.print("res:");
        Ribbon rib = new Ribbon();
        System.out.print(rib.solution(A,k));
    }
}

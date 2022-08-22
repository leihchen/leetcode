package OA;

import java.util.HashMap;
import java.util.Map;

//给一个array，问所有element都duplicate（出现次数＞1，比如［1，1，2，2］）的subarray有多少
public class duplicateSubarray {
    public int solution(int[] array){
        Map<Integer, Integer> memo = new HashMap<>();
        for(int i=0; i<array.length; i++){
            if(!memo.containsKey(array[i])){
                memo.put(array[i], 0);
            }
        }

        int right = 0;
        int left = 0;
        int res = 0;
        int n = array.length;
        while(right<n){
            memo.put(array[right], memo.get(array[right])+1);
            right++;
            while(duplicateCheck(memo)){
                res++;
                memo.put(array[left], memo.get(array[left])-1);
                left++;
            }
        }

        return res;

    }

    private boolean duplicateCheck(Map<Integer, Integer> map){
        for(int value: map.values()){
            if(value<2){
                return false;
            }
        }
        return true;
    }

    public static void main(String[] args){
        duplicateSubarray ds = new duplicateSubarray();
        int[] array = new int[]{1,1,1,3};
        System.out.print(ds.solution(array));
    }
}

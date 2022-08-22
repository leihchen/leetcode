package OA;
//给一个数“456”，看里面有多少个可以被3整除，比如“4”，“5，”，“6”，“45”， “56”，“456”

import java.util.HashSet;
import java.util.Set;

public class numDivide {
    public int solution(String num){
        if(num == null || num.length() == 0) return 0;
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for(int i=0; i<num.length(); i++){
            for(int j=i+1; j<=num.length(); j++){
                String val = num.substring(i, j);
                int temp = Integer.parseInt(val);
                if(!set.contains(temp) && temp%3==0){
                    res++;
                }
                set.add(temp);
            }
        }
        return res;
    }

    public static void main(String[] args){
        String num = "999";
        numDivide nd = new numDivide();
        System.out.print(nd.solution(num));
    }
}

//找到出现频率最多的数字, 升序输出, [1,1,2,3,4,4] => [1,4]

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class increaseNum {
    public int[] solution(int[] array){
        Map<Integer, Integer> mapping = new HashMap<>();
        for(int i:array){
            mapping.put(i, mapping.getOrDefault(i,0)+1);
        }
        int max = Integer.MIN_VALUE;
        for(int i: mapping.values()){
            max = Math.max(i,max);
        }
        List<Integer> list = new ArrayList<>();
        for(int i: mapping.keySet()){
            if(mapping.get(i) == max){
                list.add(i);
            }
        }
        int[] res = new int[list.size()];
        for(int i=0; i<res.length; i++){
            res[i] = list.get(i);
        }
        return res;
    }

    public static void main(String[] args){
        int[] array = {1,1,1,4,4,4};
        increaseNum in = new increaseNum();
        int[] res = in.solution(array);
        for(int i: res){
            System.out.print(i);
            System.out.print(" ");
        }
    }
}

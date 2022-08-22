package OA;
//一个没有重复数字的数组，将这个数组所有相邻的数，打乱顺序生成一个int[2]放进一个array里面, 再打乱顺序给你，让你返回原数组。(倒序也是正确的)
//例子:
//[3, 5, 1, 2, 4]
//有相邻pair [[3, 5], [5, 1], [1, 2], [2, 4]]
//打乱顺序后为[[4, 2], [1, 2], [5, 3], [5, 1]]

import java.util.HashMap;
import java.util.Map;

public class ShuffleThePieces {
    public int[] solution(int[][] arraies, int[] path){
        Map<Integer, Integer> map = new HashMap<>();
        Map<Integer, Integer> mapB = new HashMap<>();
        for(int i=0; i<path.length; i++){
            map.put(path[i], 0);
            mapB.put(path[i], 0);
        }
        for(int[] array: arraies){
            if(map.get(array[1])==1 || mapB.get(array[0]) == 1){
                int temp = array[1];
                array[1] = array[0];
                array[0] = temp;
            }
            map.put(array[1],1);
            mapB.put(array[0],1);
        }
        int[] res = new int[path.length];
        Map<Integer, Integer> trace = new HashMap<>();
        for(int[] array: arraies){
            trace.put(array[0], array[1]);
        }
        int start = 0;
        for(int key: map.keySet()){
            if(map.get(key) == 0){
                start = key;
            }
        }

        int index = 0;
        while (trace.containsKey(start)){
            res[index] = start;
            start = trace.get(start);
            index++;
        }
        res[index] = start;
        return res;
    }

    public static void main(String[] args){
        int[][] arries = new int[][]{{3,1}, {5,3}, {5,7}, {7,4},{9,4}};
        int[] path = new int[]{1,3,5,7,4,9};
        ShuffleThePieces st = new ShuffleThePieces();
        int[] res = st.solution(arries, path);
        for(int i=0; i<res.length; i++){
            System.out.print(res[i]);;
            System.out.print(" ");
        }
    }
}

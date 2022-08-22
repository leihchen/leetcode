package test;//给一个array和一个matrix。 matrix里面每一个vector<int>的形式必定是[l,r,target]，固定只有3个数。 然后要求统计array里 index从l 到 r这个区间出现了多少次target这个数。 比如:
//array = [1,1,2,3,2]
//matrix = [[1,2,1],
//[2,4,2],
//[0,3,1]] output : 5


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class matrixOverQueries {
    public int solution(int[][] matrix, int[] array){
        if(matrix.length == 0 || matrix == null) return 0;
        Map<Integer, List<int[]>> map = new HashMap<>();
        for(int[] i: matrix){
            if(map.containsKey(i[2])){
                int[] range = {i[0], i[1]};
                map.get(i[2]).add(range);
            }else{
                ArrayList<int[]> list = new ArrayList<>();
                int[] temp = {i[0], i[1]};

                list.add(temp);
                map.put(i[2], list);
            }
        }

        int res = 0;

        for(int i=0; i<array.length; i++){

            if(map.containsKey(array[i])){
                List<int[]> list = map.get(array[i]);
                for(int[] range: list){
                    System.out.print(range[0]);
                    System.out.print("->");
                    System.out.print(range[1]);
                    System.out.print("\n");
                    if(i>=range[0] && i<=range[1]){

                        res++;
                    }
                }
            }
        }
        return res;


    }

    public static void main(String[] args){
        int[][] matrix = {{1,2,1}, {2,4,2}, {0,3,1}};
        matrixOverQueries mq = new matrixOverQueries();
        int[] array = {1,1,2,3,2};
        System.out.print("res: ");
        System.out.print(mq.solution(matrix, array));
    }
}

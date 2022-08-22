//给一个array和一个matrix。 matrix里面每一个vector<int>的形式必定是[l,r,target]，固定只有3个数。 然后要求统计array里 index从l 到 r这个区间出现了多少次target这个数。 比如:
//array = [1,1,2,3,2]
//matrix = [[1,2,1],
//[2,4,2],
//[0,3,1]] output : 5
//因为在matrix[0], array的index 1到2区​​​​​​​​​​​​​​​​​​​间出现了1 一次， matrix[1], array的index 2到4区间出现2 两次。 matrx[2], array的index 0到3区间出现1 两次
//这个题如果直接暴力解O(n*n)会有两个test case过不了。我是用hashmap<int, vector<pair<int,int>>> 。 key是target， value是index区间。 这样走一遍array，每次确定一下当前index在不在区间里就行了。
//1， [[1,2],[0,1]]
//2, [[2,4]]
//然后loop一遍array,
//i =0, arr​ = 1, 然后这个时候判断map.containKey(arr)，然后走一遍key里的value，因为 0 <= i <= 1 ，所以output++;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class matrixOverQueries {
    public int solution(int[][] matrix, int[] array){
        Map<Integer, List<int[]>>mapping = new HashMap<>();
        if(matrix.length == 0 || matrix == null) return 0;
        int res = 0;

        for(int i=0; i<matrix.length; i++){
            int[] distance = {matrix[i][0], matrix[i][1]};
            int val = matrix[i][2];
            if(!mapping.containsKey(val)){
                List<int[]> list = new ArrayList<>();
                list.add(distance);
                mapping.put(val, list);
            }else{
                mapping.get(val).add(distance);
            }
        }//将想知道的距离信息和target放入map中

        for(int i=0; i<array.length; i++){
            if(mapping.containsKey(array[i])){
                List<int[]> dis = mapping.get(array[i]);
                for(int[] nums: dis){
                    if(i>=nums[0] && i<=nums[1]){
                        res++;
                    }
                }
            }
        }//遍历一遍array，看map中有没有想找的，然后核对位置信息，符合就加一
        return res;
    }

    public static void main(String[] args){
        int[][] matrix = {{1,3,1}, {2,4,3}, {0,3,1}};
        matrixOverQueries mq = new matrixOverQueries();
        int[] array = {1,1,2,3,2};
        System.out.print("res: ");
        System.out.print(mq.solution(matrix, array));
    }
}

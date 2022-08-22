package OA;
// 给数组内的所有数字按频率排序然后从右下角交错输出，频率相等就按大小排。
//例子:
//[[2,2,3],
//[1,1,1],
//[2,2,4]]
//
//按频率排序结果:
//[3, 4,1,1,1,2,2,2,2]
//
//输出的时候从右下斜着填 (先填m[2][2], 然后m[2][1],然后m[1][2], 然后m[2][0], 然后[1][1] .... 最后m[0][0])
//[[2,2,2],
//[2,1,1],
//[1,4,3]]

import java.util.*;

public class sortmatrixByoccurance {
    public void solution(int[][] matrix){
        Map<Integer, Integer> map = new HashMap<>();
        int m = matrix.length;
        int n = matrix[0].length;
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                map.put(matrix[i][j], map.getOrDefault(matrix[i][j],0)+1);
            }
        }

        List<Map.Entry<Integer, Integer>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                if(o1.getValue() == o2.getValue()){
                    return o1.getKey().compareTo(o2.getKey());
                }else{
                    return o1.getValue().compareTo(o2.getValue());
                }
            }
        });
        int index = 0;
        for(int i=m-1; i>=0; i--){
            for(int j=n-1; j>=0; j--){
                matrix[i][j] = list.get(index).getKey();
                list.get(index).setValue(list.get(index).getValue()-1);
                if(list.get(index).getValue() == 0){
                    index++;
                }
            }
        }

    }

    public static void main(String[] args){
        sortmatrixByoccurance sb = new sortmatrixByoccurance();
        int[][] matrix = new int[][]{{2,2,3,3},{1,1,1,2},{2,2,4,4},{9,10,11,12}};
        sb.solution(matrix);
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                System.out.print(matrix[i][j]);
                System.out.print(" ");
            }
            System.out.print("\n");
        }
    }
}

package OA;

import java.util.ArrayList;
import java.util.List;

//给了一个3行m列的举证，一个3x3的sliding window，求问每个sliding window里面包不包含1-9这9这数。例子:
//1 2 3 4
//4 5 6 3
//8 7 9 2
//这个矩阵一共有二个valid的sliding  window:
//1 2 3
//4 5 6
//8 7 9
//和
//2 3 4
//5 6 3
//7 9 2
//第一个是包含的, 第二个不包含，所以output的结果是 [true]
public class matrixFind {
    public List<Boolean> solution(int[][] matrix, int m){
        int row = 0;
        int col = 0;
        List<Boolean> res = new ArrayList<>();
        while(row<=matrix.length-m){
            col = 0;
            while(col<=matrix[0].length-m){
                int i = row;
                int[] num = new int[9];
                for(; i<row+m; i++){
                    int j=col;
                    for(; j<col+m; j++){
                        num[matrix[i][j]-1]++;
                    }
                }
                boolean flag = true;
                for(int k=0; k<num.length; k++){
                    if(num[k] == 0){
                        flag = false;
                    }
                }
                res.add(flag);
                col++;
            }
            row++;
        }
        return res;
    }

    public static void main(String[] args){
        matrixFind mf = new matrixFind();
        int[][] matrix = new int[][]{{1,2,3,4,5}, {4,5,6,3,6}, {8,7,9,2,8},{1,2,3,4,5},{4,5,6,7,1}};
        List<Boolean> res = mf.solution(matrix, 3);
        for(boolean val: res){
            System.out.print(val);
            System.out.print(" ");
        }
    }
}

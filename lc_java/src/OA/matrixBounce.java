package OA;

import java.util.Arrays;

public class matrixBounce {
    public int[] solution(int[][] matrix){
        int n = matrix.length;
        int[] res = new int[n-1];
        for(int i=1; i<n; i++){
            int row = i;
            int col = 0;
            int sum = 0;
            boolean flag = false;
            while(col<n){
                sum+=matrix[row][col];
                if(!flag) {
                    row--;
                    col++;
                }else{
                    row++;
                    col++;
                }
                if(row==0){
                    flag = true;
                }

            }
            res[i-1] = sum;
        }
        Arrays.sort(res);
        return res;
    }


    public static void main(String[] args){
        matrixBounce mb = new matrixBounce();
        int[][] matrix = new int[][]{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}, {16,17,18,19,20}, {21,22,23,24,25}};
        int[] res = mb.solution(matrix);
        for(int i=0; i<res.length; i++){
            System.out.print(res[i]);
            System.out.print(" ");
        }
    }

}

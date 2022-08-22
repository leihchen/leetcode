package test;//这个题大家都没说是转K次 但是 diag不转 搞得我第一次运行发现有bug才改过来 Example:
//[[1, 2, 3],
//[4, 5, 6],
//[7, 8, 9]] -->
//[[1, 4, 3],
//[8, 5, 2],
// [7, 6, 9]]
//[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11,12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
//--> [[1,16,11,6,5], [22,7,12,9,2],
//
//[23,18,13,8,3]​​​​​​​​​​​​​​​​​​​, [24,17,14,19,4], [21,10,15,20,25]]
//
public class rotateOverDigonals {
    public void solution(int[][] matrix, int k){
        if(matrix.length == 0 || matrix == null) return ;
        k = k%4;
        for(int i=0; i<k; i++){
            rotate(matrix);
        }

    }

    public void rotate(int[][] matrix){
        int rightdiagonal = matrix.length-1;
        for(int i=0; i<matrix.length; i++){
            for(int j=i; j<matrix[0].length; j++){
                if(j!=rightdiagonal){
                    int temp = matrix[i][j];
                    matrix[i][j] = matrix[j][i];
                    matrix[j][i] = temp;
                }
            }
            rightdiagonal--;
        }
        for(int i=0; i<matrix.length; i++){
            reverse(matrix[i], 0, matrix.length-1, i, matrix.length-1-i);
        }
    }

    public void reverse(int[] matrix, int start, int end, int leftdigonal, int rightdigonal){
        if(start>=end) return ;
        for(int i=0; i<(start+end)/2; i++){
            if( i== leftdigonal || matrix.length-i-1==rightdigonal || i == rightdigonal || matrix.length-i-1 == leftdigonal) continue;
            int temp = matrix[i];
            matrix[i] = matrix[matrix.length-1-i];
            matrix[matrix.length-1-i] = temp;
        }
    }




    public static void main(String[] args){
        rotateOverDigonals rd = new rotateOverDigonals();
        int[][] matrix = {{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}};
        rd.solution(matrix, 1);
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                System.out.print(matrix[i][j]);
                System.out.print(",");
            }
            System.out.print("\n");
        }

    }
}

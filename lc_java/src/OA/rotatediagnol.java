package OA;
//这个题大家都没说是转K次 但是 diag不转 搞得我第一次运行发现有bug才改过来 Example:
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

public class rotatediagnol {
    public void solution(int[][] matrix, int k){
        if(matrix == null || matrix.length == 0) return ;
        int m = matrix.length;
        int n = matrix[0].length;
        while(k>0) {
            for (int i = 0; i < m; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (i != j && i != n-1 - j) {
                        swap(matrix, i, j, j, i);
                    }
                }
            }

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n / 2; j++) {
                    if (i != j && i != n-1 - j) {
                        swap(matrix, i, j, i, n -1- j);
                    }
                }
            }
            k--;
        }

    }

    private void swap(int[][] matrix, int i, int j, int x, int y){
        int temp = matrix[i][j];
        matrix[i][j] = matrix[x][y];
        matrix[x][y] = temp;
    }

    public static void main(String[] args){
        rotatediagnol rd = new rotatediagnol();
        int[][] matrix = new int[][]{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15},{16,17,18,19,20},{21,22,23,24,25}};
        rd.solution(matrix, 61);
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                System.out.print(matrix[i][j]);
                System.out.print(" ");
            }
            System.out.print("\n");
        }

    }


}

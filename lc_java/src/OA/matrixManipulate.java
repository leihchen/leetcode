package OA;

public class matrixManipulate {
    public void solution(int[][] matrix, int[] action){
        for(int i=0; i<action.length; i++){
            if(action[i] == 0){
                rotate(matrix);
            }else if(action[i]==1){
                flip1(matrix);
            }else if(action[i]==2){
                flip2(matrix);
            }
        }
    }


    private void rotate(int[][] matrix){
        int m = matrix.length;
        int n = matrix[0].length;
        for(int i=0; i<m; i++){
            for(int j=i+1; j<n; j++){
                swap(matrix,i ,j, j, i);
            }
        }

        for(int i=0; i<m; i++){
            for(int j=0; j<n/2; j++){
                swap(matrix, i, j, i, n-1-j);
            }
        }
    }

    private void flip1(int[][] matrix){
        for(int i=0; i<matrix.length; i++){
            for(int j=i+1; j<matrix[0].length; j++){
                swap(matrix, i, j, j, i);
            }
        }
    }

    private void flip2(int[][] matrix){
        int n = matrix.length;
        int end = matrix[0].length-1;
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<=end; j++){
                swap(matrix, i, j, n-1-j, n-1-i);
            }
            end--;
        }
    }

    private void swap(int[][]matrix, int x, int y, int i, int j){
        int temp = matrix[x][y];
        matrix[x][y] = matrix[i][j];
        matrix[i][j] = temp;
    }

    public static void main(String[] args){
        int[][] matrix = new int[][]{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
        int[] action = new int[]{2,2,1,1,0,0,0,0};
        matrixManipulate mm = new matrixManipulate();
        mm.solution(matrix, action);
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                System.out.print(matrix[i][j]);
                System.out.print(" ");
            }
            System.out.print("\n");
        }
    }

}

package OA;
//"." empty
//"#" box
//"*" obstacle
//"...##.....*#."
//"...####......"
//rotate後變成
//..
//..
//..
//##
//##
//#.
//#.
//..
//..
//..
//.*
//.#
//..
//然後
//box會往下掉
//obstacle則不動
//output:
//..
//..
//..
//..
//..
//..
//..
//..
//.#
//##
//#*
//#.
//##
public class obmatrix {
    public char[][] solution(char[][] matrix){
        int m = matrix.length;
        int n = matrix[0].length;
        char[][] res = new char[n][m];
        for(int i=0; i<m; i++){
            int end = n-1;
            for(int j=n-1; j>=0; j--){
                if(matrix[i][j] == '#'){
                    matrix[i][j] = '.';
                    matrix[i][end] = '#';
                    end--;
                }else if(matrix[i][j] == '*'){
                    end = j-1;
                }
            }
        }

        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                res[i][j] = matrix[j][i];
            }
        }

        for(int i=0; i<n; i++){
            for(int j=0; j<m/2; j++){
                char temp = res[i][j];
                res[i][j] = res[i][m-j-1];
                res[i][m-j-1] = temp;
            }
        }

        return res;
    }

    public static void main(String[] args){
        char[][] matrix = new char[][]{{'.','.','#','.','*','#','.','.'},{'#','.','*','#','#','.','.','.'},{'.','#','#','.','.','.','.','#'},{'.','.','.','#','#','*','#','.'}};
        obmatrix ob = new obmatrix();
        char[][] res = ob.solution(matrix);
        for(int i=0; i<res.length; i++){
            for(int j=0; j<res[0].length; j++){
                System.out.print(res[i][j]);
                System.out.print(" ");
            }
            System.out.print("\n");
        }
    }
}

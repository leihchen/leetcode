import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Diagnolsort {

    public void solution(int[][] matrix){
        if(matrix.length == 0 || matrix == null) return ;
        int n = matrix[0].length;
        int m = matrix.length;
        for(int i=0; i<n; i++){
            int row = 0;
            int col = i;
            List<Integer> list = new ArrayList<>();
            while(row<m && col<n){
                list.add(matrix[row][col]);
                row++;
                col++;
            }
            Collections.sort(list);
            row = 0;
            col = i;
            for(int num : list){
                matrix[row][col] = num;
                row++;
                col++;
            }
        }

        for(int j=0; j<m; j++){
            int row = j;
            int col = 0;
            List<Integer> list = new ArrayList<>();
            while(row<m && col<n){
                list.add(matrix[row][col]);
                row++;
                col++;
            }
            Collections.sort(list);
            row = j;
            col = 0;
            for(int num : list){
                matrix[row][col] = num;
                row++;
                col++;
            }

        }


    }

    public static void main(String[] args){
        Diagnolsort ds = new Diagnolsort();
        int[][] matrix = {{8,4,1,7}, {4,4,1,3}, {4,8,9,2}, {2,5,3,1}};

        System.out.print("original: ");
        System.out.print("\n");
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                System.out.print(matrix[i][j]);
                System.out.print(" ");
            }
            System.out.print("\n");
        }
        ds.solution(matrix);
        System.out.print("res: ");
        System.out.print("\n");
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                System.out.print(matrix[i][j]);
                System.out.print(" ");
            }
            System.out.print("\n");
        }

    }
}

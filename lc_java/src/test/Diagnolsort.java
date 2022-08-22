package test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Diagnolsort {

    public void solution(int[][] matrix){
        if(matrix.length == 0|| matrix==null) return ;
        for(int i=0; i<matrix[0].length; i++){
            int row = 0;
            int col = i;
            List<Integer> list = new ArrayList<>();
            while(row<matrix.length && col<matrix[0].length){
                list.add(matrix[row][col]);
                row++;
                col++;
            }
            Collections.sort(list);
            row = 0;
            col = i;
            int index = 0;
            while(row<matrix.length && col<matrix[0].length){
                matrix[row][col] = list.get(index);
                index++;
                row++;
                col++;
            }

        }
        for(int i=0; i<matrix.length; i++){
            int row = i;
            int col = 0;
            List<Integer> list = new ArrayList<>();
            while(row<matrix.length && col<matrix[0].length){
                list.add(matrix[row][col]);
                row++;
                col++;
            }
            Collections.sort(list);
            row = i;
            col = 0;
            int index = 0;
            while(row<matrix.length && col<matrix[0].length){
                matrix[row][col] = list.get(index);
                index++;
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

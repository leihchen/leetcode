package OA;

import test.brokeKeyboard;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

//给定一个nxn的二维矩阵，这个矩阵会有一层层的“边框”，第一个边框是从(0,0)到(0,n-1)到(n-1,n-1)到(n-1,0)，
//第二个是从(1,1)到(1,n-2)到(n-2,n-2)到(n-2,1)，诸如此类，对于每一个边框，
//把位于边框上的元素排序，按照从左上开始的顺时针顺序重新放在边框上。例如[ [4,1],[2,3] ]会变成[[1,2],[3,4]]
public class bordersort {
    public void solution(int[][] matrix){

        int m = matrix.length;
        int n = matrix[0].length;
        List<PriorityQueue<Integer>> temp = new ArrayList<>();

        int rowstart = 0;
        int rowend = m-1;
        int colstart = 0;
        int colend = n-1;
        while(rowstart<=rowend && colstart<=colend){
            PriorityQueue<Integer> pq = new PriorityQueue<>();
            for(int i=colstart; i<=colend; i++){
                pq.offer(matrix[rowstart][i]);
            }
            rowstart++;
            for(int i=rowstart; i<=rowend; i++){
                pq.offer(matrix[i][colend]);
            }
            colend--;
            if(rowstart<=rowend) {
                for (int i = colend; i >= colstart; i--) {
                    pq.offer(matrix[rowend][i]);
                }
                rowend--;
            }

            if(colstart<=colend) {
                for (int i = rowend; i>=rowstart; i--){
                    pq.offer(matrix[i][colstart]);
                }
                colstart++;
            }
            temp.add(pq);
        }

        rowstart = 0;
        rowend = m-1;
        colstart = 0;
        colend = n-1;

        int j = 0;
        while(rowstart<=rowend && colstart<=colend){
            PriorityQueue<Integer> pq = temp.get(j);
            for(int i=colstart; i<=colend; i++){
                matrix[rowstart][i]=pq.poll();
            }
            rowstart++;
            for(int i=rowstart; i<=rowend; i++){
                matrix[i][colend]=pq.poll();
            }
            colend--;
            if(rowstart<=rowend) {
                for (int i = colend; i >= colstart; i--) {
                    matrix[rowend][i]=pq.poll();
                }
                rowend--;
            }

            if(colstart<=colend) {
                for (int i = rowend; i>=rowstart; i--){
                    matrix[i][colstart]=pq.poll();
                }
                colstart++;
            }
            j++;
        }



    }

    public static void main(String[] args){
        bordersort bs = new bordersort();
        int[][] matrix = new int[][]{{7,3,4,9,55},{6,8,10,30,33},{5,11,12,15,77},{16,20,77,888,909},{87,62,54,30,21}};
        bs.solution(matrix);
        for(int[] ma: matrix){
            for(int i=0; i<ma.length; i++){
                System.out.print(ma[i]);
                System.out.print(", ");
            }
            System.out.print("\n");
        }
    }
}

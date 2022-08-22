package OA;

import java.util.PriorityQueue;

public class blackwhiteMatrix {
    void solution(int[][] matrix, int[][] queries){
        if(matrix == null || matrix.length == 0) return ;
        for(int[] query: queries){
            int starti = query[0];
            int startj = query[1];
            int m = query[2];
            PriorityQueue<Integer> black = new PriorityQueue<>();
            PriorityQueue<Integer> white = new PriorityQueue<>();
            for(int i=starti; i<starti+m; i++){
                for(int j=startj; j<startj+m; j++){
                    if((i+j)%2==0){
                        black.offer(matrix[i][j]);
                    }
                    if((i+j)%2==1){
                        white.offer(matrix[i][j]);
                    }
                }
            }

            for(int i=starti; i<starti+m; i++){
                for(int j=startj; j<startj+m; j++){
                    if((i+j)%2==0){
                        matrix[i][j] = black.poll();
                    }
                    if((i+j)%2==1){
                        matrix[i][j] = white.poll();
                    }
                }
            }
        }
    }

    public static void main(String[] args){
        blackwhiteMatrix bm = new blackwhiteMatrix();
        int[][] matrix = new int[][]{{83,67,39,85,11,21,87}, {25,48,74,7,15,74,90}, {13,10,87,57,3,75,36}, {19,47,89,48,16,7,81}, {79,40,68,70,25,59,96}};
        bm.solution(matrix, new int[][]{{0,0,3},{0,3,4}});
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                System.out.print(matrix[i][j]);
                System.out.print(" ");
            }
            System.out.print("\n");
        }
    }
}

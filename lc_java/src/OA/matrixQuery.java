package OA;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
//deactivate一行或一列找最小值
public class matrixQuery {
    public int[] solution(int[][] queries, int m, int n){
        int[] memo = new int[2];
        int minval = Integer.MAX_VALUE;
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if((i+1)*(j+1)<minval){
                    memo[0] = i;
                    memo[1] = j;
                    minval = (i+1)*(j+1);
                }
            }
        }
        Set<Integer> row = new HashSet<>();
        Set<Integer> col = new HashSet<>();
        List<Integer> res = new ArrayList<>();
        for(int[] query: queries){
            if(query.length==1){
                if(!row.contains(memo[0]) && !col.contains(memo[1])){
                    res.add(minval);
                }else{
                    minval = Integer.MAX_VALUE;
                    for(int i=0; i<m; i++){
                        if(row.contains(i)) continue;
                        for(int j=0; j<n; j++){
                            if(col.contains(j)) continue;
                            if((i+1)*(j+1)<minval){
                                memo[0] = i;
                                memo[1] = j;
                                minval = (i+1)*(j+1);
                            }
                        }
                    }
                    res.add(minval);
                }
            }else{
                if(query[0] == 1){
                    row.add(query[1]-1);
                }else if(query[0] == 2){
                    col.add(query[1]-1);
                }
            }
        }

        int[] result = new int[res.size()];
        for(int i=0; i<res.size(); i++){
            result[i] = res.get(i);
        }
        return result;
    }

    public static void main(String[] args){
        matrixQuery mq = new matrixQuery();
        int[][] queries = new int[][]{{0},{1,2},{0},{2,1},{0},{1,1},{0}};
        int[]res = mq.solution(queries, 3,4);
        for(int i=0; i<res.length; i++){
            System.out.print(res[i]);
            System.out.print(" ");
        }
    }
}

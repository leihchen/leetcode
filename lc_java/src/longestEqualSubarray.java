//int fun(int[] a), a 由 1 和 0 组成. 求 0，1 个数相同的 subarray 最大长度.
//用 HashMap 就好 (convert 0 to -1) rameWindow (有点忘记名字了) : Given an int n, print
//the *** window frame of the number;
//Example: input -> n = 6
//output -> [
//"********", --> 8 *
//"* *", -> 2 * 加 六个 ' ' (space)
//"* *",
//"* *",
//"* *",
//"********"
// ]

public class longestEqualSubarray {
    public char[][] solution(int n){
        char[][] res  =new char[n][n+2];
        for(int i=0; i<n; i++){
            if(i==0 || i==n-1){
                for(int j=0; j<n+2; j++){
                    if(j==0||j==n+1){
                        res[i][j] = '*';//首尾添加*号
                    }else{
                        res[i][j] = ' ';
                    }
                }
            }else{
                for(int j=0; j<n+2; j++){
                    res[i][j] = '*';
                }
            }

        }
        return res;
    }


    public static void main(String[] args){
        longestEqualSubarray ls = new longestEqualSubarray();
        char[][] res = ls.solution(6);
        for(int i=0; i<res.length; i++){
            for(int j=0; j<res[0].length; j++){
                System.out.print(res[i][j]);
            }
            System.out.print("\n");
        }
    }

}

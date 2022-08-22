import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//Input: int[][] query = [ [-1,3], [-5,-3], [3,5] , [2,4], [-3,-2], [-1,4], [5,5] ]
//output: 3
//题目是说，query里面的每个int[] -> 代表的是left & right。你要做的是，找到最小的number个数，使得它们都可以包含在所有的int[] 里面。
//比如：对于这个query来说，numbers -》-3，3，5。所以，最小的Min number就是3。
//[-5,5],
public class findMinimumNumber {
    public int solution(int[][] arrays){
        if(arrays.length == 0 || arrays==null) return 0;

        Arrays.sort(arrays,(a,b)->(a[0]-b[0]));
        int left = arrays[0][0];
        int right = arrays[0][1];
        List<int[]> res = new ArrayList<>();
        int con = 0;
        for(int i=1; i<arrays.length; i++){
            if(arrays[i][0]<=right){
                left = arrays[i][0];
                right  = Math.min(right, arrays[i][1]);//若左边小于前面的右，取最小区间

            }else{
                int[] temp = {left, right};
                res.add(temp);
                left = arrays[i][0];
                right = arrays[i][1];
            }
        }
        int[] temp = {left, right};
        res.add(temp);


//        for(int i=0; i<res.size(); i++){
//            int[]range = res.get(i);
//            System.out.print("(");
//            System.out.print(range[0]);
//            System.out.print(",");
//            System.out.print(range[1]);
//            System.out.print(")");
//            System.out.print(" ");
//
//        }
        return res.size();//结果就是有几个这种交集区间

    }

    public static void main(String[] args){
        int[][] query = {{-1,3}, {-5,-3}, {3,5}, {2,4}, {-3,-2}, {-1,4}, {5,5}};
        findMinimumNumber fm = new findMinimumNumber();
        int consequence = fm.solution(query);
        System.out.print("res: ");
        System.out.print(consequence);
    }

}

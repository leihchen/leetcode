//Give an array and find the count of a pair number and a single number combination in a row of this array. Target array is a[i - 1], a, a[i + 1]
//Example:
//Input: a = [1, 1, 2, 1, 5, 3, 2, 3]
//Output: 3
//Explain:
//[1, 1, 2] -> two 1 and one 2(O)
//[1, 2, 1] -> two 1 and one 2(O)
//[2, 1, 5] -> one 2, one 1 and one five(X)
//[1, 5, 3] -> (X)
//[5, 3, 2] -> (X)
//[3, 2, 3] -> (O)
//int result = 0;
//if (nums.length <= 2) {
//return 0; }
//int pre1 = nums[1];
//int pre2 = nums[0];
//for (int i = 2; i < nums.length; i++) {
//if (nums[i] != pre1 && nums[i] != pre2 && pre1 != pre2) { result++;
//}
//pre2 = pre1; pre1 = nums[i];
//}
public class GoodTuple {

    public int solution(int[] array){
        if(array.length == 0 || array==null) return 0;
        int pre2 = array[0];
        int pre1 = array[1];
        int res = 0;
        int total = array.length-2;
        for(int i=2; i<array.length; i++){
            if((array[i] == pre1 && pre1 != pre2) || (array[i] == pre2 && array[i] != pre1) || (pre1 == pre2 && array[i] != pre1)){//防止三个相等
                res++;//三个三个判断
            }
            pre2 = pre1;
            pre1 = array[i];
        }
        return res;
    }

    public static void main(String[] args){
        GoodTuple gt = new GoodTuple();
        int[] array = {1,1,2,1,1,3,2,3};
        System.out.print(gt.solution(array));
    }

}

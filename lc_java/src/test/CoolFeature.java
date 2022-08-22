package test;//最开始有3个case超时，原因是每次执行问加起来等于target的情况有多少种的那个query时，我都是 重新算一遍记录b数组每个数出现次数的HashMap，其实只用最开始构好，然后每次做第二种 query([index, num])，更新一下HashMap就行。改了就过了。
//Give three array ​a​, ​b​ and ​query​. This one is hard to explain. Just read the example.
//Input:
//a = [1, 2, 3]
//b = [3, 4]
//query = [[1, 5], [1, 1, 1], [1, 5]]
//Output:
//[2, 1]
//Explain:
//Just ignore every first element in sub-array in the query.
//So we will get a new query like this query = [[5], [1, 1], [5]]
//Only record the result when meet the single number in new query array.
//And the rule of record is find the sum of the single number.
//The example above is 5 = 1 + 4 and 5 = 2 + 3, there are two result.
//So currently the output is [2]
//When we meet the array length is larger than 1, such as [1, 1]. That means we will replace the b[x] = y, x is the first element, y is second element. So in this example, the b will be modify like this b = [1, 4]
//And finally, we meet the [5] again. So we will find sum again. This time the result is 5 = 1 + 4. So currently the output is [2, 1]
//note: Don't have to modify the query array, just ignore the first element.
//Time:
//Function findSum is O(a * b)
//Function modifyArrayb is O(1)
//Function treverse is O(query)
//So total maybe O(a * b * query)
//I think this problem must has some better solution, but I am almost run out of time.

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CoolFeature {

    public List<Integer> solution(int[] a, int[] b, int[][] query){
        if(query.length == 0 || query == null) return new ArrayList<>();
        Map<Integer, Integer> map = new HashMap<>();
        for(int i: a){
            map.put(i, map.getOrDefault(i, 0)+1);
        }
        List<Integer> res = new ArrayList<>();
        Map<Integer, Integer> memo = new HashMap<>();
        for(int i=0; i<query.length; i++){
            int[] temp = query[i];
            if(temp.length == 3){
                b[temp[1]] = temp[2];
            }else{
                int sum = query[i][1];
                int count = 0;
                for(int j=0; j<b.length; j++){
                    int target = sum-b[j];
                    if(map.containsKey(target)){
                        count+=map.get(target);
                    }
                }
                res.add(count);
            }
        }
        return res;

    }



    public static void main(String[] args){
        CoolFeature cf = new CoolFeature();
        int[] A = {1,1,2,3};
        int[] B = {1,1};
        int[][] query = {{1,5}, {1,0,1}, {1,5},{1,7}};
        System.out.print(cf.solution(A,B, query));
    }
}


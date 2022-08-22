import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//Divide Array into two with equal size, make sure every element in each two array is unique. return empty list if impossible.
//先sort再分配，连续超过2，
// 就不能满足了
// 我是一个HashMap记录原数组每个数出现次数，
// 如果有某个超过2的话，就不可能满足要求直接返回 空;
// 如果出现次数都是2或以下的话，就对出现两次的平分到两个数组，剩下只出现一次的随便分给两 个数组就行只要保障最后两个数组长度一致。
public class equalsizeuniqueArray {
    public List<List<Integer>> solution(int[] array){
        if(array.length == 0 || array.length%2 != 0) return new ArrayList<>();
        List<List<Integer>> res  =new ArrayList<>();
        List<Integer> list1 = new ArrayList<>();
        List<Integer> list2 = new ArrayList<>();
        Arrays.sort(array);//先排序
        for(int i=0; i<array.length; i++){
            if(i!=0 && array[i] == array[i-1]){
                if(!list2.contains(array[i])){
                    list2.add(array[i]);
                }else{
                    return new ArrayList<>();//如果出现三次直接return
                }
            }else{
                if(list1.size()<array.length/2){
                    list1.add(array[i]);//先加list1
                }else{
                    list2.add(array[i]);
                }
            }
        }
        res.add(list1);
        res.add(list2);
        return res;

    }

    public static void main(String[] args){
        int[] array = {1,2,1,0,0,5};
        equalsizeuniqueArray eq = new equalsizeuniqueArray();
        List<List<Integer>> res = eq.solution(array);
        if(res.isEmpty()){
            System.out.print("no such array");
            return ;
        }
        List<Integer> list1 = res.get(0);
        List<Integer> list2 = res.get(1);
        System.out.print("res: ");
        for(int i:list1){
            System.out.print(i);
            System.out.print(" ");
        }
        System.out.print("\n");
        System.out.print("res2: ");
        for(int i:list2){
            System.out.print(i);
            System.out.print(" ");
        }
        System.out.print("\n");

    }
}

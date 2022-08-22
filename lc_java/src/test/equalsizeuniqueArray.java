package test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//Divide Array into two with equal size, make sure every element in each two array is unique. return empty list if impossible.

public class equalsizeuniqueArray {
    public List<List<Integer>> solution(int[] array){
        if(array.length == 0 || array == null || array.length%2 != 0) return new ArrayList<>();
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> l1 = new ArrayList<>();
        List<Integer> l2 = new ArrayList<>();
        Arrays.sort(array);
        l1.add(array[0]);
        for(int i=1; i<array.length; i++){
            if(array[i-1] == array[i]){
                if(l2.contains(array[i])) return new ArrayList<>();
                else{
                    l2.add(array[i]);
                }
            }else{
                if(l1.size()<array.length/2){
                    l1.add(array[i]);
                }else{
                    l2.add(array[i]);
                }
            }
        }
        res.add(l1);
        res.add(l2);
        return res;


    }

    public static void main(String[] args){
        int[] array = {1,2,1,0,0,5,7,8};
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

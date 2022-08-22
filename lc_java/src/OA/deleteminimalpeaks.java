package OA;
//delete minimal peaks，minimal peaks定义是array里左右都比自己小的element，然后找所有minimal peaks中最小的，删掉，加到一个list，直到删完。
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

public class deleteminimalpeaks {
    public List<Integer> solution(int[] array){
        List<Integer> list = new ArrayList<>();
        List<Integer> res = new ArrayList<>();
        for(int num: array){
            list.add(num);
        }

        while(!list.isEmpty()){
            if(list.size() == 1){
                res.add(list.get(0));
                break;
            }
            int index = 0;
            int val = Integer.MAX_VALUE;
            for(int i=0; i<list.size(); i++){
                if(i==0){
                    if(list.get(i)>=list.get(i+1)){
                        if(list.get(i)<=val){
                            index = i;
                            val = list.get(i);
                        }
                    }
                }else if(i==list.size()-1){
                    if(list.get(i)>=list.get(i-1)){
                        if(list.get(i)<=val){
                            index = i;
                            val = list.get(i);
                        }
                    }
                }else{
                    if(list.get(i)>=list.get(i-1) && list.get(i)>=list.get(i+1)){
                        if(list.get(i)<=val){
                            index = i;
                            val = list.get(i);
                        }
                    }
                }
            }
            list.remove(index);
            res.add(val);

        }
        return res;
    }

    public static void main(String[] args){
        deleteminimalpeaks dm = new deleteminimalpeaks();
        int[] array = new int[]{1,2};
        System.out.print(dm.solution(array));
    }
}

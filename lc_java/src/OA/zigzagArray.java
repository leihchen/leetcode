package OA;
//给出一个由W,D和L组成的字符串和一个初始为空的答案字符串，按照以下操作序列来操，先判断字符串中是否有W，有的话从原字符串末尾。中删除一个W，加到答案字符串
//再判断字符串中是否有D，有的话删一个D，加到答案字符串末尾，然后对L进行同样的判断。不断循环这三个步骤直到原字符串为空，返回答案字符串。
public class zigzagArray {
    public boolean[] solution(int[] array){
        int n = array.length;
        boolean[] res = new boolean[n-2];
        for(int i=2; i<n; i++){
            int a = array[i-2];
            int b = array[i-1];
            int c = array[i];
            if((b>a && b>c) || (a>b && b<c)){
                res[i-2] = true;
            }
        }
        return res;
    }

    public static void main(String[] args){
        int[] array = new int[]{1,2,1,3,1,2,3,2,3,1};
        zigzagArray za = new zigzagArray();
        boolean[] res = za.solution(array);
        for(int i=0; i<res.length; i++){
            System.out.print(res[i]);
            System.out.print(" ");
        }
    }
}

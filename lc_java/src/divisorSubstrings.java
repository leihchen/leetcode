//给一个数字, 和一个 k 值(表示除数位数),
// 看能用多少个sub number整除 ex: n = 1220 k = 2 => 1220 % 12 != 0, 1220 % 22 != 0, 1220 % 20 == 0 => ans : 1

public class divisorSubstrings {
    public int solution(int num, int k){
        if(num == 0) return 0;
        int res = 0;
        String val = String.valueOf(num);
        for(int i=0; i<=val.length()-k; i++){
            String subnum = val.substring(i,i+k);//截取子数
            int sub_num = Integer.parseInt(subnum);
            System.out.print(sub_num);
            System.out.print(" ");
            if(sub_num==0) continue;//防止除数为0
            if(num%sub_num == 0) res++;
        }
        return res;
    }

    public static void main(String[] args){
        int num = 1220010203;
        int k =2;
        divisorSubstrings ds = new divisorSubstrings();
        int res = ds.solution(num, k);
        System.out.print("res: ");
        System.out.print(res);
    }
    //将数字转为字符然后截取
}

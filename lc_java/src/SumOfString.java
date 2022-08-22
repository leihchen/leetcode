

import java.util.Arrays;
import java.util.List;

//String fun(String a, String b) a 和 b数字组成, a和b的第ith个数字依次相加, 返回新String
//eg. a = "99" b = "1" return "910" 如果写Java的话最好用StringBuil​​​​​​​​​​​​​​​​​​​der, String 会 TLE
public class SumOfString {
    public String solution(String a, String b){
        if(a==null || b==null) return "";
        StringBuilder sb = new StringBuilder();
        if(a.length()>b.length()){
            while(a.length() != b.length()){
                b = "0"+b;
            }

        }
        if(a.length()<b.length()){
            while(a.length() != b.length()){
                a = "0"+a;//将位数补充完整
            }
        }

        for(int i=0; i<a.length(); i++){
            int num1 = Character.getNumericValue(a.charAt(i));
            int num2 = Character.getNumericValue(b.charAt(i));
            int sum = num1+num2;//一位一位相加
            sb.append(sum);
        }
        return sb.toString();

    }
    public static void main(String[] args){
       String a = "1";
       String b = "10";
       SumOfString ss = new SumOfString();
       String res = ss.solution(a,b);
       System.out.print("res: ");
       System.out.print(res);
    }
}

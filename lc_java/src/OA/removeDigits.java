package OA;
//給兩個字串 a, b
//求從字串b刪掉一個數字以後 a字串小於b的可能有幾種 (alphanumeric sort)
//例如: a="d1c" b = "1b2z"
//則可能有b = "b2z"
//及 b = "1bz"
//兩種情況做比較
public class removeDigits {
    public int solution(String a, String b){
        if(a==null || a.length() == 0 || b==null || b.length() == 0) return 0;
        int res = 0;
        for(int i=0; i<b.length(); i++){
            if(i==0){
                if(b.charAt(i)>='0' && b.charAt(i)<='9'){
                    String newstring = b.substring(1);
                    if(newstring.compareTo(a)>0){
                        res++;
                    }
                }
            }else if(i==b.length()-1){
                if(b.charAt(i)>='0' && b.charAt(i)<='9'){
                    String newstring = b.substring(0, b.length()-1);
                    if(newstring.compareTo(a)>0){
                        res++;
                    }
                }
            }else{
                if(b.charAt(i)>='0' && b.charAt(i)<='9'){
                    String newstring = b.substring(0, i)+b.substring(i+1);
                    if(newstring.compareTo(a)>0){
                        res++;
                    }
                }
            }
        }
        return res;
    }

    public static void main(String[] args){
        String a = "ab12c";
        String b = "1zz456";
        removeDigits rd = new removeDigits();
        System.out.print(rd.solution(a, b));
    }
}

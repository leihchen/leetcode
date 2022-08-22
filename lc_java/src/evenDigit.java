//Find how many numbers have even digit in a list.
// Ex.Input: A = [12, 3, 5, 3456]
// Output: 2
public class evenDigit {
    public int solution(int[] A){
        int res = 0;
        if(A.length==0) return res;
        for(int num: A){
            while(num>0){
                int temp = num%10;
                if(temp %2 == 0){
                    res++;
                    break;
                }else{
                    num = num/10;
                }
            }
        }
        return res;
    }
    public static void main(String[] args){
        evenDigit ed = new evenDigit();
        int[] A = {12,344,52,3456};
        System.out.print("Res: ");
        System.out.print(ed.solution(A));
    }
}

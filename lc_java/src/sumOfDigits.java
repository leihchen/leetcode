//8. 花式数位求和
//25631 -> 2 - 5 + 6 - 3 +1


public class sumOfDigits {
    public int solution(long num){
        int res = 0;
        boolean flag = false;
        while(num != 0){
            if(!flag) {
                res += num % 10;
                flag = true;
            }else{
                res -= num % 10;
                flag = false;
            }
            num = num/10;
        }
        return res;
    }
    public static void main(String[] args){
        long num = 123456789;
        sumOfDigits sd = new sumOfDigits();
        System.out.print("res: ");
        System.out.print(sd.solution(num));
    }
}

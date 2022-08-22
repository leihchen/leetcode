//给一个数num, 返回这个数每一个digit的乘积: product 和 每一位digit的和: sum的差值

public class numQuestion {
    public int solution(int num){
        if(num == 0) return 0;
        int product = 1;
        int sum = 0;
        while(num >0){
            int temp = num%10;
            product*=temp;
            sum+=temp;
            num = num/10;
        }
        return Math.abs(product-sum);
    }

    public static void main(String[] args){
        numQuestion nq = new numQuestion();
        System.out.print(nq.solution(0));
    }
}

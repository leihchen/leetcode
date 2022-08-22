package OA;
// n = 123456, reverse后 214365
//
public class reverseDigitspair {
    public int solution(int digit){
        if(digit<10) return digit;
        String str = String.valueOf(digit);
        char[] string = str.toCharArray();
        int i=0;
        while(i<string.length-1){
            char temp = string[i];
            string[i] = string[i+1];
            string[i+1] = temp;
            i=i+2;
        }
        String newstring = new String(string);
        return Integer.parseInt(newstring);
    }

    public static void main(String[] args){
        reverseDigitspair rd = new reverseDigitspair();
        System.out.print(rd.solution(12));

    }
}

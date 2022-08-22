//反转连续的两个字符, "abcdef" => "badcfe"

public class stringReverse {
    public String solution(String word){
        if(word.length() ==0 || word == null) return "";
        char[] wordArray = word.toCharArray();
        for(int i=0; i<wordArray.length-1; i++){
            if((wordArray[i+1]-'a')-(wordArray[i]-'a')==1){//判断是否连续
                char temp = wordArray[i+1];
                wordArray[i+1] = wordArray[i];
                wordArray[i] = temp;
            }
        }
        return new String((wordArray));
    }


    public static void main(String[] args){
        stringReverse sr = new stringReverse();
        String res = sr.solution("bacdef");
        System.out.print(res);

    }
}

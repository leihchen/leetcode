package test;//这道题我看之前各位的面经一直对什么地方该split什么地方不该感到困惑，结果还是让我遇到了，仔 细看题之后发现只需要按照空格分就行了，
//broken keyboard 键盘的部分英文字母键坏了(注意只有字母键坏了) 给定一个String 和 一个char Array(没坏的字母键)，输出String中能打出的字符串数。
//栗子:
//input “hello, world!” ['i','e','o','l','h']; output: 1 (只能打出 hello 这个单词)
//input “5 + 3 = 8” []; output: 5 (没有英文字母， 5， +， 3， =， 8 都可以打出) 之前面经有过的题。
// 输入一组words和一组valid letters，判断有多少个words是valid。
// 判断条件是 words里的所有upper and lower letter必须在valid letters里面。
// 如果word里面有special character不用 管。注意valid letter只有小写，但是words里面有大写的也算valid。
// 比如words = [hEllo##, This^^], valid letter = [h, e, l, 0, t, h, s]; "hello##" 就是valid，因为h，e，l，o都在valid letter 里面，
// “This^^” 不 valid, 因为i不在valid letter里面

import java.util.Arrays;
import java.util.List;

public class brokeKeyboard {
    public int solution(String sentence, List<Character> list){
        if(sentence == null || list.isEmpty()) return 0;
        String[] words = sentence.toLowerCase().split(" ");
        boolean flag = false;
        int res = 0;
        for(String word: words){
            for(Character c: word.toCharArray()){
                if(Character.isLowerCase(c)){
                    if (!list.contains(c)) {
                        flag = true;
                        break;
                    }

                }
            }
            if(flag == true){
                flag = false;
            }else{
                res++;
            }

        }
        return res;
    }

    public static void main(String[] args){
        brokeKeyboard bk = new brokeKeyboard();
        String sentence = "hEllo##, This^^";
        List<Character> list = Arrays.asList('i','e','o','l','h','t', 's');
        int res = bk.solution(sentence, list);
        System.out.print("res: ");
        System.out.print(res);
    }
}



//之前面经也出现过。compare两个string，只有小写字母。 每个stirng内部可以任意换位置，
// 所以 位置不重要。每个string内部两个letter出现的频率也可以互换，
// 所以这题只需要两个string每个 frequency出现的次数要一样。
//比如“babzccc” 和 “bbazzcz” 就返回“true”，因为z和c可以互换频率。 但是“babzcccm” 和 “bbazzczl” 就不一样，因为m在第一个里出现过，第二个里没有出现过。

import java.util.HashSet;
import java.util.Set;

public class stringCompare {
    public boolean solution(String str1, String str2){
        int[] letter1 = new int[26];
        int[] letter2 = new int[26];
        if(str1.length() != str2.length()) return false;
        if(str1.length() == 0 || str2.length() == 0) return false;
        Set<Character> set = new HashSet<Character>();
        for(int i=0; i<str1.length(); i++){
            set.add(str1.charAt(i));
        }
        for(int i=0; i<str2.length(); i++){
            if(!set.contains(str2.charAt(i))) return false;//判断字母是否相同
        }


        for(int i=0; i<str1.length(); i++){
            letter1[str1.charAt(i)-'a']++;
            letter2[str2.charAt(i)-'a']++;
        }

        int[] frequency1 = new int[str1.length()];//记录字母出现的频率
        int[] frequency2 = new int[str2.length()];
        for(int i=0; i<26; i++){
            if(letter1[i] != 0){
                frequency1[letter1[i]]++;//记录frequency的一个趋势
            }
            if(letter2[i] != 0){
                frequency2[letter2[i]]++;
            }
        }

        for(int i=0; i<frequency1.length; i++){
            if(frequency1[i] != frequency2[i]) return false;//判断frequency是否一样
        }

        return  true;
    }

    public static void main(String[] args){
        String str1 = "babzcccm";
        String str2 = "bbazzczl";
        stringCompare str = new stringCompare();
        System.out.print(str.solution(str1, str2));


    }
}

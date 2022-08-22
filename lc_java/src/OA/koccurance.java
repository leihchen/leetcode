package OA;
//
public class koccurance {
    public int[] solution(String[] words, String sequence){
        int[] res = new int[words.length];
        for(int i=0; i<words.length; i++){
            int num = 0;
            String word = words[i];
            while(sequence.indexOf(word) != -1){
                word+=words[i];
                num++;
            }
            res[i] = num;
        }
        return res;
    }

    public static void main(String[] args){
        koccurance ko = new koccurance();
        String[] words = new String[]{"ab", "babc", "bca"};
        int[] res = ko.solution(words, "aaa");
        for(int i=0; i<res.length; i++){
            System.out.print(res[i]);
            System.out.print(" ");
        }
    }
}

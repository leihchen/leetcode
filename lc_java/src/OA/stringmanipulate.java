package OA;
//给出一个由W,D和L组成的字符串和一个初始为空的答案字符串，按照以下操作序列来操，先判断字符串中是否有W，有的话从原字符串中删除一个W，加到答案字符串末尾。
//再判断字符串中是否有D，有的话删一个D，加到答案字符串末尾，然后对L进行同样的判断。不断循环这三个步骤直到原字符串为空，返回答案字符串。
public class stringmanipulate {
    public String solution(String string){
        String res = "";
        while(!string.isEmpty()){
            if(string.indexOf('W')!=-1){
                int index = string.indexOf('W');
                string = string.substring(0, index)+string.substring(index+1);
                res+="W";
            }
            if(string.indexOf('D')!=-1){
                int index = string.indexOf('D');
                string = string.substring(0, index)+string.substring(index+1);
                res+="D";
            }

            if(string.indexOf('L')!=-1){
                int index = string.indexOf('L');
                string = string.substring(0, index)+string.substring(index+1);
                res+="L";
            }
        }
        return res;

    }

    public static void main(String[] args){
        String str = "WWLLDDDLLLWWWDDWDWDWDDDDLLLWW";
        stringmanipulate sm = new stringmanipulate();
        System.out.print(sm.solution(str));
    }
}

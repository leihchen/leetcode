package OA;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//给出两种操作(0,a,b)和(1,a,b)，前者表示给你一个axb的矩形，后者是一次询问，询问你当前收到的所有矩形是否都能放进一个axb的盒子里，
// 并不是问能否同时塞进盒子。而是只要矩形1能放进去，矩形2也能放进，就算OK。
public class matrixadapt {
    public boolean[] solution(int[][] queries){
        int size1 = 0;
        int size0 = 0;
        for(int[] query: queries){
            if(query[0] == 1){
                size1++;
            }else if(query[0] == 0){
                size0++;
            }
        }
        boolean[] res = new boolean[size1];
        if(size1!=0 && size0 == 0){
            Arrays.fill(res, true);
            return res;
        }
        int i=0;
        int maxlength = 0;
        int maxwidth = 0;
        size1 = 0;
        size0 = 0;
        for(int[] query: queries){
            int length = Math.max(query[1], query[2]);
            int width = Math.min(query[1], query[2]);
            if(query[0]==0){
                maxlength = Math.max(maxlength, length);
                maxwidth = Math.max(maxwidth, width);
                size0++;
            }else{
                if(size1>=0 && size0 == 0){
                    res[i] = true;
                }else if(length>maxlength || width>maxwidth){
                    res[i] = false;
                }else if(length<=maxlength && width<=maxwidth){
                    res[i] = true;
                }
                i++;
                size1++;
            }
        }
        return res;
    }

    public static void main(String[] args){
        matrixadapt ma = new matrixadapt();
        int[][] array = new int[][]{{1,5,6},{1,6,7},{0,6,7},{1,6,7},{0,4,3},{1,6,7},{1,7,8}};
        boolean[] res = ma.solution(array);
        for(int i=0; i<res.length; i++){
            System.out.print(res[i]);
            System.out.print(" ");
        }
    }
}



//给int n, m，想象n*m的矩阵M, M[i,j] = (i+1)*(j+1)，0-based ​​​​​​​​​​​​​​​​​​​一系列query，有三种类型，第一种是查询矩阵中最小的元素，第二、三分别是禁用某一行、列。 一个2D array的min number的query
//题目是说给你一个2d array。其中array​[j] = (i+1)*(j+1)。这个给定。 然后给一堆query，有三种不同的格式:
//第一种是让你返回当前array中的最小值
//第二种是让你把某一行disable
//第三种是把某一列disab​​​​​​​​​​​​​​​​​​​le
//当然disable了之后最小值就不能用了

import java.util.ArrayList;
import java.util.List;

public class matrixSearch {
    private int m;
    private int n;
    private List<Integer> col;
    private List<Integer> row;
    public matrixSearch(int m, int n){
        this.m = m;
        this.n = n;
        col = new ArrayList<>();
        row = new ArrayList<>();
    }

    public int querymin(){
        int min = Integer.MAX_VALUE;
        for(int i=0; i<m; i++){
            if(row.contains(i)) continue;
            for(int j=0; j<n; j++){
                if(col.contains(j)) continue;
                if((i+1)*(j+1)<min){
                    min = (i+1)*(j+1);
                }
            }
        }
        return min;
    }

    public void ban_col(int val){
        if(val<n && val>=0) {
            col.add(val);
        }
    }

    public void ban_row(int val){
        if(val>=0 && val<m) {
            row.add(val);
        }
    }

    public static void main(String[] args){
        matrixSearch ms = new matrixSearch(2,3);
        System.out.print(ms.querymin());
        ms.ban_col(0);
        System.out.print(ms.querymin());
        ms.ban_col(1);
        System.out.print(ms.querymin());
    }
}

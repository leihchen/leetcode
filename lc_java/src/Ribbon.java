//Maximum size of ribbon
//Given a list representing the length of ribbon, and the target number "k" parts of ribbon.
//we want to cut ribbon into k parts with the same size, at the same time we want the maximum size.
//Ex.
//Input: A = [1, 2, 3, 4, 9], k = 5 Output: 3

public class Ribbon {
    public int solution(int[] A, int k){
        int start = 1;
        int max = 0;
        for(int i:A){
            if(i>max){
                max = i;
            }
        }
        int end = max;
        while(start+1<end){
            int mid = start+(end-start)/2;
            if(sum(A, mid)>k){
                start=mid;
            }else{
                end=mid;
            }

        }
        if(sum(A, end) == k) return end;
        if(sum(A, start) == k) return start;
        return 0;

    }

    public int sum(int[] A, int cut){
        int sum = 0;
        for(int i:A){
            sum+=i/cut;
        }
        return sum;//计算size
    }

    public static void main(String[] args){
        int[] A = {1,2,3,4,9};
        int k=5;
        System.out.print("res:");
        Ribbon rib = new Ribbon();
        System.out.print(rib.solution(A,k));
    }
}

package OA;

import java.util.ArrayList;
import java.util.List;

public class justifyNewspaperText {
    public static void main(String[] args) {
        justifyNewspaperText test = new justifyNewspaperText();
        String[][] lines = {{"hello", "world"}, {"How", "areYou", "doing"}, {"Please look", "and align", "to right", "OK?"}};
        String[] aligns = {"LEFT", "RIGHT", "RIGHT"};
        int width = 16;
        String[] res = test.justifyNewspaperText(lines, aligns, width);
        for (String x : res) {
            System.out.println(x);
        }
    }

    public static final String STAR = "*";
    public static final String SPACE = " ";
    public static final String[] POS = {"LEFT", "RIGHT"};

    public String[] justifyNewspaperText(String[][] lines, String[] aligns, int width) {
        String bar = repeat(STAR,width + 2);
        List<String> output = new ArrayList<>();
        output.add(bar);
        for (int i = 0; i < lines.length; ++i) {
            String[] line = lines[i];
            List<StringBuilder> sbs = new ArrayList<>();
            sbs.add(new StringBuilder());
            int curSb = 0;
            sbs.get(curSb).append(line[0]);
            for (int j = 1; j < line.length; ++j) {
                String word = line[j];
                if (sbs.get(curSb).length() + word.length() + 1 <= width) {
                    sbs.get(curSb).append(SPACE).append(word);
                } else {
                    sbs.add(new StringBuilder());
                    curSb++;
                    sbs.get(curSb).append(word);
                }
            }
            for (StringBuilder sb : sbs) {
                output.add(getLine(sb, aligns[i], width));
            }
        }
        output.add(bar);
        String[] res = new String[output.size()];
        for (int i = 0; i < res.length; ++i) {
            res[i] = output.get(i);
        }
        return res;
    }

    public String getLine(StringBuilder sb, String pos, int width) {
        int remainingSpace = width - sb.length();
        String res = STAR;
        if (pos.equals(POS[0])) {
            res += sb.toString() + repeat(SPACE, remainingSpace) + STAR;
        } else {
            res += repeat(SPACE, remainingSpace) + sb.toString() + STAR;
        }
        return res;
    }
    private String repeat(String str, int time){
        String newstring = "";
        for(int i=0; i<time; i++){
            newstring+=str;
        }
        return newstring;
    }
}

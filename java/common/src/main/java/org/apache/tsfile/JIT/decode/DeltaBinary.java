package org.apache.tsfile.JIT.decode;

public class DeltaBinary {
    // each round has 8 vectors
    public static short[][] read_in_bytes = new short[65][8];
    public static short[][] end = new short[65][8];
    public static short[][] pos_bias = new short[65][8];
    public static short[]   max_read_bytes = new short[65];
//    public static short[]   pos_adds = new short[65];

    static {
        for(int width = 1; width <= 64; width++) {
            int pos = 0, e = 0, bias = 0;
            for(int i = 0;i<8;i++) {
                e = bias + width;
                int count;
                if ((e & 0x7) == 0) count = e / 8 - 1;
                else count = e / 8;
                pos_bias[width][i] = (short) pos;
                read_in_bytes[width][i] = (short) (count + 1);
                end[width][i] = (short) e;
                max_read_bytes[i] = (short) Math.max(max_read_bytes[i], (short) (count + 1));
                bias = e % 8;
            }
        }
    }
}

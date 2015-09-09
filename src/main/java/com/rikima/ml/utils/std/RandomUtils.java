package com.rikima.ml.utils.std;

import java.util.*;

public class RandomUtils {
    long seed = 20071026;
    Random rand;

    public RandomUtils(long seed) {
        this.seed = seed;
        this.rand = new Random(seed);
    }

    public int randomInt(int base) {
        if (base == 1) {
            return 0;
        }
        int r = (int)(rand.nextDouble()*base);
        return r;
    }
}

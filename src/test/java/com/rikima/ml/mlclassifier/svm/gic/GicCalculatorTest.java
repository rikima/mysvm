package com.rikima.ml.mlclassifier.svm.gic;

import junit.framework.TestCase;

import java.io.*;

import com.rikima.ml.mlclassifier.svm.gic.GicCalculator;

public class GicCalculatorTest extends TestCase {
    PrintStream stderr = System.err;

    public void testGicCalsC100() throws Exception {
        String fname = "./data/test.svmdata.100";
        double c = 1.0e-5;

        String[] args = {"-i", String.format("%s", fname), "-c", String.format("%f", c)};
        {
            GicCalculator.main(args);
        }
    }

    public void testTest() throws Exception {
        assertEquals(1, 1);
    }
}

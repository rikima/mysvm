package com.rikima.ml.mlclassifier.svm.gic;

import com.rikima.ml.mlclassifier.svm.gic.CvExecutor;
import junit.framework.TestCase;

import java.io.*;

public class CvExecutorTest extends TestCase {


    static PrintStream stderr = System.err;
    static PrintStream stdout = System.out;

    public void testReset() throws Exception {
        String fname = "./data/test.svmdata.50";
        double c = 1.0;

        String[] args = {"-i", String.format("%s", fname), "-c", String.format("%f", c)};

        // init 
        {
            stderr.println("# init case");
            CvExecutor.main(args);
        }
    }
}
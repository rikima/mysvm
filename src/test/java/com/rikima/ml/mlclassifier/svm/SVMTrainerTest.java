package com.rikima.ml.mlclassifier.svm;

import com.rikima.ml.mlclassifier.svm.SVMTrainer;
import junit.framework.TestCase;

public class SVMTrainerTest extends TestCase {

    public void testMain2() throws Exception {

    	String fname = "./data/w1a/w1a.svmdata";
    	double c = 1.0;

    	String[] args = {"-i",String.format("%s", fname), "-c", String.format("%f", c) };
        
    	SVMTrainer.main(args);
    }
    
	
    public void testMain() throws Exception {
    	
    	String fname = "./data/a1a/a1a.svmdata";
    	double c = 1.0;
    	
    	String[] args = {"-i",String.format("%s", fname), "-c", String.format("%f", c) };
        
    	SVMTrainer.main(args);
    }
    
	
    public void testTest() throws Exception {
        assertEquals(1, 1);
    }
}

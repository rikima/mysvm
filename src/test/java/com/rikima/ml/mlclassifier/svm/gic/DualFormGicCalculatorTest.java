package com.rikima.ml.mlclassifier.svm.gic;

import junit.framework.TestCase;

import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;
import com.rikima.ml.mlclassifier.svm.SVMTrainer;
import com.rikima.ml.mlclassifier.svm.gic.DualFormGicCalculator;
import com.rikima.ml.mlclassifier.svm.gic.GicCalculator;

import java.io.*;

public class DualFormGicCalculatorTest extends TestCase {
    PrintStream stderr = System.err;
    PrintStream stdout = System.out;

    public void testEstimate() throws Exception {
        double c = 1.0;
        String fname = "data/w1a/w1a.svmdata";
        double epsilon = 1.0e-4;

        epsilon /= c;

        MLDataFactory mf = new MLDataFactory(fname);
        MLData mldata = mf.getWithIndexor();

        SVMTrainer trainer = new SVMTrainer(mldata, c);
        
        GicCalculator self = new GicCalculator(trainer, epsilon);
        self.estimate();
    }


    public void testNormalTrain() throws Exception {
        double c = 1.0;
        String fname = "data/w1a/w1a.svmdata";
        double epsilon = 1.0e-4;


        MLDataFactory mf = new MLDataFactory(fname);
        MLData mldata = mf.getWithIndexor();

        SVMTrainer trainer = new SVMTrainer(mldata, c);
        
        DualFormGicCalculator self = new DualFormGicCalculator(trainer, epsilon);
        
        assertNotNull(self);

        self.normalTrain();
        
        int psv = self.trainer.getUnitModel().positiveSVs();
        int nsv = self.trainer.getUnitModel().negativeSVs();
        int bpsv = self.trainer.getUnitModel().positiveBSVs();
        int bnsv = self.trainer.getUnitModel().negativeBSVs();
        
        assertEquals(51, psv);
        assertEquals(103, nsv);
        
        assertEquals(24, bpsv);
        assertEquals(8, bnsv);
    }

    public void testScore() throws Exception {
        double c = 1.0;
        String fname = "data/w1a/w1a.svmdata";
        double epsilon = 1.0e-5;


        MLDataFactory mf = new MLDataFactory(fname);
        MLData mldata = mf.getWithIndexor();

        SVMTrainer trainer = new SVMTrainer(mldata, c);
        
        DualFormGicCalculator self = new DualFormGicCalculator(trainer, epsilon);
        
        assertNotNull(self);

        self.normalTrain();
        
        FeatureVector wv = self.trainer.getWeightVector();
        assertNotNull(wv);
        
        // test 0
        {
        	int i = 0;
            FeatureVector fv = self.trainer.getMLData().getCExample(i);
            double s = wv.dot(fv) - self.trainer.getUnitModel().bias();
            
            double ks = self.score(fv);
            
            stdout.println("  score:" + s);
            stdout.println("k score:" + ks);
        	
            assertEquals(s, ks, 1.0e-5);
        }

        
        // test 10
        {
        	int i = 10;
            FeatureVector fv = self.trainer.getMLData().getCExample(i);
            double s = wv.dot(fv) - self.trainer.getUnitModel().bias();
            
            double ks = self.score(fv);
            
            stdout.println("  score:" + s);
            stdout.println("k score:" + ks);
        	
            assertEquals(s, ks, 1.0e-5);
        }

        
        // test 100
        {
            int i = 100;
            FeatureVector fv = self.trainer.getMLData().getCExample(i);
            double s = wv.dot(fv) - self.trainer.getUnitModel().bias();
            
            double ks = self.score(fv);
            
            stdout.println("  score:" + s);
            stdout.println("k score:" + ks);

            assertEquals(s, ks, 1.0e-5);
        }

        // test 1000
        {
            int i = 1000;
            FeatureVector fv = self.trainer.getMLData().getCExample(i);
            double s = wv.dot(fv) - self.trainer.getUnitModel().bias();
            
            double ks = self.score(fv);
            
            stdout.println("  score:" + s);
            stdout.println("k score:" + ks);

            assertEquals(s, ks, 1.0e-5);
        }
    }

    public void testTest() throws Exception {
        assertEquals(1, 1);
    }
}

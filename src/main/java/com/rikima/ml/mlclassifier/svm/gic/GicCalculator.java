package com.rikima.ml.mlclassifier.svm.gic;

import java.io.*;
import java.util.Arrays;

import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;
import com.rikima.ml.mlclassifier.svm.SVMTrainer;

public class GicCalculator extends LOO {
    static boolean DEBUG = true;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;
    
    // fields ----------
    
    static boolean useInitReset = true;
    
    static double EPS = 1.0e-10;
    static double epsilon = 1.0e-4;
    
    //private static double c = Double.NaN;
    // constructors ----
    /**
     * constructor
     * 
     */
    GicCalculator(SVMTrainer t, double e) {
        super(t);
        epsilon = e;
    }
    
    // methods ---------
    /**
     * estimate gic
     * 
     * @param epsilon
     * @return
     */
    public double estimate() {
        this.normalTrain();
        double loss = likellihood();
        assert loss > 0;
        
        double bias = 0;

        FeatureVector m = null;

        int size = trainer.getMLData().size();
        for (int i = 0;i < size;++i) {
            // acc 
            CategoricalFeatureVector cfv = trainer.getMLData().getCExample(i);

            boolean isSv = false;
            try {
                if (Math.abs(alphas[i]) < EPS) {
                    m = new FeatureVector(this.originalWeightVector);
                    stderr.println("#" + i + " : Non SV");
                } else {
                    trainByVariationalDistribution(i);
                    m = getWeightVector();
                    isSv = true;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            if (isSv) {
                double[] pdl = this.partialDeferentialLogLikelihood(i);
                double insBias = influenceFunction(m).dot(pdl);
                
                stderr.println(" insBIas=" + insBias);
                bias += insBias;
            }
            
            if (DEBUG) {
                stderr.println(" bias=" + bias);
            }
            
            double f = score(m, cfv) * cfv.getClassValue();
            double y = cfv.getClassValue();
            if (DEBUG) {
                double insProb = prob(f);
                double insLoss = loss(f);

                stderr.println("y=" + y + " f(=y*(wx+b))=" + f + " prob=" + insProb + " loss=" + insLoss);
            }
            
            this.acc.set(f, y);
        }
        

        bias /= trainer.getMLData().size();

        // 
        stdout.println("#  GIC=" + (loss+bias));
        stdout.println("# loss=" + loss);
        stdout.println("# bias=" + bias);
        
        stdout.println(" " + acc.toString());
        
        return bias;
    }
    
    
    /**
     * calc influence function
     * 
     * @param m
     * @param trainer
     * @return
     */
    FeatureVector influenceFunction(FeatureVector m) {
        try {
            m = m.plus(this.originalWeightVector, -1);
            m.times(1.0 / epsilon);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // debug
        stderr.println("l2 of m=" + m.l2norm());
                
        return m;
    }
    
    /**
     * 
     * 
     * @param idx
     * @param trainer
     * @return
     */
    double[] partialDeferentialLogLikelihood(int idx) {
        if (buf == null) {
            buf = new double[trainer.featureDimension()];
        }
        Arrays.fill(buf, 0.0);
        
        CategoricalFeatureVector cfv = trainer.getMLData().getCExample(idx);

        // score =  w * x + b
        // f = y (w * x + b)
        double f = super.score(super.originalWeightVector, cfv) * cfv.getClassValue();  
        double insProb = prob(f);

        for (int j = 0;j < cfv.size();++j) {
            double g = cfv.valueByIndex(j) * cfv.getClassValue() * (1.0 - insProb);
            buf[cfv.idByIndex(j)-1] = g;
        }
        return buf;
    }

    // main ----
    /**
     * main
     *
     * @param args
     */
    public static void main(String[] args) {
        String fname = null;

        for (int i = 0;i < args.length;++i) {
            if (args[i].equals("-e") || args[i].equals("--epsilon")) {
                epsilon = Double.parseDouble(args[++i]);
            } else if (args[i].equals("-c")) {
                c = Double.parseDouble(args[++i]);
            } else if (args[i].equals("-i") || args[i].equals("--input")) {
                fname = args[++i];
            }
        }

        epsilon /= c;
        
        if (fname == null) {
            stderr.println("please check options: -i [fname] -c [c value] -e [epsilon value");
            System.exit(1);
        }

        try {
            MLDataFactory mf = new MLDataFactory(fname);
            MLData mldata = mf.getWithIndexor();

            SVMTrainer trainer = new SVMTrainer(mldata, c);
            
            GicCalculator self = new GicCalculator(trainer, epsilon);
            self.estimate();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

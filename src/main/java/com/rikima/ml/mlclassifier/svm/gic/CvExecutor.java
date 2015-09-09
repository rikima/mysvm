package com.rikima.ml.mlclassifier.svm.gic;

import java.io.*;

import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;
import com.rikima.ml.mlclassifier.svm.SVMTrainer;
import com.rikima.ml.utils.std.ArrayUtils;

public class CvExecutor extends LOO {
    static boolean DEBUG = true;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;
    
        
    // constructors ----
    
    /**
     * constructor
     * 
     */
    CvExecutor(SVMTrainer t) {
        super(t);
    }
    
    // methods ---------
    
    protected void reset(int i) {
        //super.trainer.getUnitModel().reset(super.alphas, i);
        super.trainer.getUnitModel().init();
    }
        
    
    /**
     * 
     */
    public void setLooDistribution(MLData mldata, int idx) {
        
        this.trainer.getUnitModel().targetIndex = idx;
        
        /*
        this.trainer.getUnitModel().targetWeight = 0.0;
        this.trainer.getUnitModel().untargetWeight = 1.0;
        */
        
        for (int i = 0;i < mldata.size();++i) {
            double w = 1.0;
            if (i == idx) {
                w = 0;
            }
            this.trainer.getUnitModel().getCs()[i] = w;
        }
    }
    
    
    /**
     * estimate LOO cv
     * 
     * @param epsilon
     * @return
     */
    public double estimate() {
    	this.normalTrain();
        double lk = trainer.likellihood();
    	assert lk > 0;
        
    	stderr.println("#likellihood=" + lk);
    	
    	FeatureVector m = null;
    	
    	int size = trainer.getMLData().size();
    	
    	int pp = 0;
    	int pn = 0;
    	int np = 0;
    	int nn = 0;
    	
    	for (int i = 0;i < size;++i) {
            CategoricalFeatureVector cfv = trainer.getMLData().getCExample(i);

            if (Math.abs(alphas[i]) > EPS) {
                try {
                    trainByVariationalDistribution(i);
                    m = getWeightVector();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                
                // debug
                ArrayUtils.println(System.err, this.trainer.getUnitModel().getAlphas());
            } else {
                m = this.originalWeightVector;
            }

            double y = score(m, cfv) * cfv.getClassValue();
            
            acc.set(y, cfv.getClassValue());
            
            /*
            if (cfv.getClassValue() > 0) {
                if (y >= 0) {
                    pp++;
                }
                else {
                	pn++;
                }
            }
            else {
            	if (y >= 0) {
            		np++;
            	}
            	else {
            		nn++;
            	}
            }
            */
    	}
            	
    	/*
        double acc = (double)(pp+nn)/size;
        stdout.println("# acc=" + acc + " (" + (pp+nn) +  "/" + size + ")");
        stdout.println("# pp pn np nn=" + pp + " " + pn + " " + np + " " + nn);
        */
    	
    	stdout.println(acc.toString());
    	
    	return acc.accuracy();
    }

    // main ----
    /**
     * main
     *
     * @param args
     */
    public static void main(String[] args) {
        double c = 1.0;
        String fname = null;

        for (int i = 0;i < args.length;++i) {
            if (args[i].equals("-c")) {
                c = Double.parseDouble(args[++i]);
            } else if (args[i].equals("-i") || args[i].equals("--input")) {
                fname = args[++i];
            }
        }

        if (fname == null) {
            stderr.println("please check options: -i [fname] -c [c value]");
            System.exit(1);
        }

        try {
            MLDataFactory mf = new MLDataFactory(fname);
            MLData mldata = mf.getWithIndexor();

            SVMTrainer trainer = new SVMTrainer(mldata, c);

            CvExecutor self = new CvExecutor(trainer);
            self.estimate();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}

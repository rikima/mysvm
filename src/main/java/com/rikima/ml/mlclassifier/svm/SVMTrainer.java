package com.rikima.ml.mlclassifier.svm;

import java.io.*;
import java.util.*;

import com.rikima.ml.mlclassifier.AbstractTrainer;
import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;
import com.rikima.ml.mlclassifier.svm.kernel.factory.KernelParams;

public class SVMTrainer extends AbstractTrainer {
    static boolean DEBUG = false;
    static PrintStream stderr = System.err;
    static PrintStream stdout = System.out;

    static String SUFFIX = ".svm";

    // fields -------------------
    boolean useWeightCollection = false;

    private MLData mldata;

    private KernelParams kParams;
    private SmoOptimizer optimizer;

    private int[] binaryLabels;

    private double c;

    // constructros --------------

    /**
     * constructor
     *
     */
    public SVMTrainer(MLData mldata, double c) {
        this(mldata, c, new KernelParams());
    }

    /**
     * constructor
     *
     * @param mldata
     * @param c
     * @param kParams
     */
    public SVMTrainer(MLData mldata, double c, KernelParams kParams) {
        this.mldata = mldata;
        
        this.binaryLabels = new int[mldata.size()];
        for (int i = 0;i < mldata.size();++i) {
            binaryLabels[i] = (int)mldata.getCExample(i).getClassValue();
        } 

        this.kParams = kParams;
        this.optimizer = new SmoOptimizer(mldata,kParams, c);
        this.c = c;

        if (useWeightCollection) {
            weightCollection();
        }
    }
    
    // methods --------------
    /**
     * weight collection
     */
    protected void weightCollection() {
        double p = 0;
        double n = 0;
        for (int i = 0;i < mldata.size();++i) {
            CategoricalFeatureVector cfv = mldata.getCExample(i);

            if (cfv.getClassValue() > 0) {
                p += 1.0;
            } else {
                n += 1.0;
            }
        }

        double w = n/p;
        w = 1.0;
        for (int i = 0;i < mldata.size();++i) {
            CategoricalFeatureVector cfv = mldata.getCExample(i);

            if (cfv.getClassValue() > 0) {
                cfv.setWeight(w);
            }
        }
    }

    
    /**
     * return c
     */
    public double getC() {
		return c;
	}

    /**
     * return likelihood
     *
     */
	public double likellihood() {
        throw new UnsupportedOperationException();
    }


    /**
     * eval loss function and gradient
     *
     */
    public double eval(double[] input, double[] gradient) {
        throw new UnsupportedOperationException();
    }
    
    /**
     * return prob
     *
     * @param cfv
     * @param wv
     * @return
     */
    public static double prob(CategoricalFeatureVector cfv, FeatureVector wv) {
        return Double.NaN;
    }

    /**
     * return gradient value
     *
     * @param idx
     * @param grad
     */
    public void getGradientElement(int idx, FeatureVector wv, double[] grad) {
        throw new UnsupportedOperationException();
    }

    /**
     * train via minimizer
     *
     */
    /*
    public FeatureVector train() {
        this.optimizer.optimize();
        return createWeightVector();
    }
    */
    public void train() {
        this.optimizer.optimize();
    }

    public FeatureVector getWeightVector() {
        return createWeightVector();
    }

    /**
     * return svm unit model
     * 
     * @return
     */
    public SVMUnitModel getUnitModel() {
        return this.optimizer.getUnitModel();
    }

    /**
     * return weight vector
     * 
     * @return
     */
    public FeatureVector createWeightVector() {
        TreeMap<Integer, Double> tm = new TreeMap<Integer, Double>();
        
        SVMUnitModel m = this.optimizer.getUnitModel();
        
        for (int i = 0;i < mldata.size();++i) {
            if (m.alpha(i) != 0.0) {
                CategoricalFeatureVector cfv = mldata.getCExample(i);

                for (int j = 0;j < cfv.size();++j) {
                    int fid = cfv.idByIndex(j);
                    double fval = cfv.valueByIndex(j);
                    
                    if (!tm.containsKey(fid)) {
                        tm.put(fid, 0.0);
                    }

                    //double v = m.alpha(i) * cfv.getClassValue() * fval; //  * binaryLabels[i];
                    double v = m.alpha(i) * binaryLabels[i] * fval; 
                    tm.put(fid, v + tm.get(fid));
                }
            }
        }
        
        return new FeatureVector(0, tm);
    }
    
    
    /**
     * return mldata
     */
    public MLData getMLData() {
    	return mldata;
    }
    
    public void reset(FeatureVector wv) {
        // TODO
    }
   
    public void init() {
        try {
            this.optimizer.setBinaryLabels(binaryLabels);
            this.optimizer.init();
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    
    public double[] getGradient() {
        return null;
    }
    
    // main --------------
    /**
     * @param args
     */
    public static void main(String[] args) {
        
        if (args.length < 4) {
            stderr.println("please input -i [file name] -c [c param] (-o [output model])");
            System.exit(1);
        }
        
        String fname = null;
        String model = null;
        double c = 1.0;
        for (int i = 0;i < args.length;++i) {

            if (args[i].equals("-i")) {
                fname = args[++i];
            } else if (args[i].equals("-c")) {
                c = Double.parseDouble(args[++i]);
            } else if (args[i].equals("-m")) {
                model = args[++i];
            }
        }

        if (model == null) {
            model = fname + SUFFIX;
        }
        
        MLDataFactory mf = new MLDataFactory(fname);
        MLData mldata = null;
        try {
            mldata = mf.getWithIndexor();
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        SVMTrainer self = new SVMTrainer(mldata, c);
        
        self.init();
        self.train();
        FeatureVector wv = self.getWeightVector();
        
        try {
            self.outputWeightVector(model, wv);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

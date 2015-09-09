package com.rikima.ml.mlclassifier.svm.gic;

import java.io.*;

import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.svm.SVMTrainer;
import com.rikima.ml.utils.std.ArrayUtils;


abstract public class LOO {
    static boolean DEBUG = true;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;
    
    static double EPS = 1.0e-10;
    
    // fields --
    
    public SVMTrainer trainer;
    public FeatureVector originalWeightVector;
    
    protected static double c;
    public double[] alphas;
    
    
    public static double epsilon = 1.0e-4;
            
    public double[] buf;
    
    protected boolean useInitReset = true;
    
    // inner class ------
    
    class Accuracy {
        int pp;
        int pn;
        int np;
        int nn;
        
        double r;
        double p;
        double f1;
        double acc;

        /**
         *
         * @param f = y * (w * x + b)
         * @param y
         */
        void set(double f, double y) {
            // relavant
            if (f >= 0) {
                // positive
                if (y > 0) {
                    pp++;
                } else {
                    nn++;
                }
            } else {
                // false negative
                if (y > 0) {
                    pn++;
                } else {
                    np++;
                }
            }
        }
        
        double recall() {
            this.r = 0;
            if (pp > 0) {
                this.r = (double)pp / (pn + pp);
            }
            return this.r;
        }
        
        double precision() {
            this.p = 0;
            if (this.pp > 0) {
                this.p = (double)pp / (pp + np);
            }
            return this.p;
        }
        
        double f1() {
            this.f1 = 0;
            if (this.p * this.r > 0) {
                this.f1 = 2.0 * this.r * this.p / (this.r + this.p);
            }
            return f1;
        }
        
        double accuracy() {
            this.acc = (double)(pp+nn) / (pp+nn+pn+np);
            return this.acc;
        }
        
        public String toString() {
            StringBuffer sb = new StringBuffer();
            
            sb.append(String.format(" acc:%f (%d / %d)\n", accuracy(), (pp+nn), (pp + pn + np + nn)));
            sb.append(String.format("   r:%f (%d / %d)\n", recall(), pp, (pp + pn)));
            sb.append(String.format("   p:%f (%d / %d)\n", precision(), pp, (pp + np)));
            
            sb.append(String.format("   f1:%f \n", f1() ));
            
            sb.append(String.format("PP PN NP NN : %d %d %d %d\n", pp, pn, np, nn));
            
            return sb.toString();
        }
    }
    
    Accuracy acc = new Accuracy();
    
    
    // constructors ----
    
    /**
     * constructor
     * 
     */
    LOO(SVMTrainer t) {
        this.trainer = t;
        
        this.alphas = new double[this.trainer.getMLData().size()];
    }
    
    // abstract methods ----
    
    /**
     * estimate
     * 
     */
    abstract double estimate();

    
    // methods ---------
    
    
    /**
     * reset for part learining
     * 
     */
    protected void reset(int i) {
        if (useInitReset) {
        	this.trainer.getUnitModel().init();
        }
        else {
            this.trainer.getUnitModel().setAlpha(this.alphas);
        }
    }
    
    /**
     * 
     * set variational distribution
     * 
     */
    protected void setLooDistribution(MLData mldata, int idx) {
        double n = mldata.size();
        
        this.trainer.getUnitModel().targetIndex = idx;

    	for (int i = 0;i < mldata.size();++i) {
            double w = (1.0 - epsilon);
            if (i == idx) {
                w += epsilon * n;
            }
            this.trainer.getUnitModel().getCs()[i] = c * w;
        }
    }
    
    /**
     * noraml train
     * 
     */
    protected void normalTrain() {
        
        stderr.print("processing normali train...");
        // normal train
        try {
            this.trainer.init();
            this.trainer.train();
            FeatureVector wv = this.trainer.getWeightVector();
            
            ArrayUtils.copy(this.trainer.getUnitModel().getAlphas(), this.alphas);
            
            this.originalWeightVector = new FeatureVector(wv);
        } catch (Exception e) {
            e.printStackTrace();	
        }
        stderr.println(" !done.\n");
    }
    
    /**
     * return score via dual form
     * 
     * 
     * @param cfv
     * @return
     */
    double score(FeatureVector cfv)  {
        double s = - this.trainer.getUnitModel().bias();
        
        for (int i = 0;i < this.trainer.getMLData().size(); ++i) {
            FeatureVector fv = this.trainer.getMLData().getCExample(i);
        }
        
        return s;
    }

    /**
     * calc score
     * 
     * @param wv
     * @param cfv
     * @return
     */
    
    double score(FeatureVector wv, FeatureVector cfv) {
        boolean useW = true;
        
        double s0 = Double.NaN;
        double s = Double.NaN;
        if (useW || DEBUG) {
            s0 = (wv.dot(cfv) - this.trainer.getUnitModel().bias());
        }
        
        if (!useW) {
            
        	double[] alpha = this.trainer.getUnitModel().getAlphas();
            MLData mldata = this.trainer.getMLData();
            
            s = - this.trainer.getUnitModel().bias();
            
            for (int i = 0;i < this.trainer.getMLData().size();++i) {
                if (Math.abs(alpha[i]) > 1.0e-5) {
                    CategoricalFeatureVector sv = mldata.getCExample(i);
                    s +=  sv.getClassValue() * alpha[i] * sv.dot(cfv);
                }
            }
        }
        
        if (!useW) {
            stderr.println("s0=" + s0 + " s=" + s);
            assert Math.abs(s0-s) < 1.0e-3;
        }
        
        return s0;
    }
    
    double prob(double f) {
    	double insProb;
        if (f < -30) {
            insProb = 0;
        } 
        else if (f > 30) {
            insProb = 1;
		} 
        else {
            double temp = 1.0 + Math.exp(-f);
            insProb = 1.0 / temp;
		}
        
        return insProb;
    }
    
    double loss(double f) {
    	double insLoss;
        if (f < -30) {
			insLoss = -f;
        } 
        else if (f > 30) {
			insLoss = 0;
        } 
        else {
            double temp = 1.0 + Math.exp(-f);
            insLoss = Math.log(temp);
        }
        
        return insLoss;
    }
    
    
    double likellihood() {
        double l = 0;
        MLData mldata = trainer.getMLData();
        for (int i = 0;i < mldata.size();++i) {
    		CategoricalFeatureVector cfv = mldata.getCExample(i);
            
            // score = ( w * x + b)
    		// f = y * (w * x + b)
    		double f = score(this.originalWeightVector, cfv) * cfv.getClassValue(); 
            
    		/*
            double insLoss, insProb;
            if (score < -30) {
    			insLoss = -score;
                insProb = 0;
            } 
            else if (score > 30) {
    			insLoss = 0;
    			insProb = 1;
    		} 
            else {
                double temp = 1.0 + Math.exp(-score);
                insLoss = Math.log(temp);
                insProb = 1.0 / temp;
    		}
            */
    		
    		double insLoss = loss(f);
            l += insLoss;
    		
    	}

    	return l;
    	
    }
    
    
    
    
    /**
     * train by variational distribution
     * 
     * @param i
     * @return
     * @throws Exception
     */
    protected void trainByVariationalDistribution(int i) throws Exception {
    	long t = System.currentTimeMillis();
    	stderr.println("-----");
    	stderr.println("#" + i + " variational train");
        
        MLData md = trainer.getMLData();
        
        setLooDistribution(md, i);
        
        // reset 
        reset(i);
        
    	// train 
        train();
        
        
        t = System.currentTimeMillis() - t;
        stderr.println("process time:" + t + " [ms]");
    }
    
    protected FeatureVector getWeightVector() {
    	return this.trainer.createWeightVector();
    }
    
    
    protected void train() {
    	this.trainer.train();
    }
}

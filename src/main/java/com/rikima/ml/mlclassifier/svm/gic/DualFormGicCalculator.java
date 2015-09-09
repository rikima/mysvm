package com.rikima.ml.mlclassifier.svm.gic;

import java.io.*;
import java.util.Arrays;

import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;
import com.rikima.ml.mlclassifier.svm.SVMTrainer;
import com.rikima.ml.mlclassifier.svm.kernel.factory.KernelParams;
import com.rikima.ml.utils.std.ArrayUtils;


public class DualFormGicCalculator extends LOO {
    static boolean DEBUG = true;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;
    
    // fields ----------
    
    static boolean useInitReset = true;
    
    static double EPS = 1.0e-10;
    static double epsilon = 1.0e-4;
    
    private double[] alphaDiff;
    
    // constructors ----
    
    /**
     * constructor
     * 
     */
    DualFormGicCalculator(SVMTrainer t, double e) {
        super(t);
        epsilon = e;
        
        this.alphaDiff = new double[t.getMLData().size()];
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

        int size = trainer.getMLData().size();
        for (int i = 0;i < size;++i) {
            
            boolean isSv = false;
            try {
                if (Math.abs(alphas[i]) < EPS) {
                    stderr.println("#" + i + " : Non SV");
                } else {
                    trainByVariationalDistribution(i);
                    isSv = true;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

            if (isSv) {
                double[] pdl = this.partialDeferentialLogLikelihood(i);
                double[] T_G = dualFormInfluenceFunction( trainer.getUnitModel().getAlphas() );
                double insBias = ArrayUtils.dot(pdl, T_G);

                
                stderr.println(" insBIas=" + insBias);
                bias += insBias;
            }
            
            if (DEBUG) {
                stderr.println(" bias=" + bias);
            }
            
            CategoricalFeatureVector cfv = trainer.getMLData().getCExample(i);

            double y = cfv.getClassValue();
            double f = y * score(cfv);

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
    private  double[] dualFormInfluenceFunction(double[] as) {
        for (int i = 0;i < as.length;++i) {
            this.alphaDiff[i] = ( as[i] - alphas[i] )  / this.epsilon;
        }
        return this.alphaDiff;
    }
    
    /**
     * return kerneled scroe
     * 
     * s = \sum_ y_j * alpha_j K(x_j, x) + b
     * 
     */
    		
    
    double score(FeatureVector fv) {
        double s = - this.trainer.getUnitModel().bias();
    	
        for (int i = 0;i < this.trainer.getMLData().size(); ++i) {
            if (alphas[i] < 1.0e-5) {
                continue;
            }
            
            CategoricalFeatureVector cfv = this.trainer.getMLData().getCExample(i);
            double y = cfv.getClassValue();
            
            s += y * alphas[i] * this.trainer.getUnitModel().getQ().getKernelValue(i, fv);
            
        }
        
        return s;
    }
    
    
    /**
     * partial differential log likelihood
     * 
     * @param idx
     * @param trainer
     * @return
     */
    private double[] partialDeferentialLogLikelihood(int idx) {
        if (buf == null) {
        	buf = new double[this.alphas.length];
        }
        Arrays.fill(buf, 0.0);
        
        CategoricalFeatureVector cfv_idx = trainer.getMLData().getCExample(idx);
    	
        // score =  w * x + b
		// f = y (w * x + b)
        double y_idx = cfv_idx.getClassValue();
        double f_idx = y_idx * score(cfv_idx);  
        
        
        double insProb = prob(f_idx);
        for (int j = 0;j < buf.length;++j) {
            CategoricalFeatureVector cfv_j = trainer.getMLData().getCExample(j);
            double y_j = cfv_j.getClassValue();
            
			double g = y_idx * (1.0 - insProb) * y_j * trainer.getUnitModel().getQ().getKernelValue(j, idx);
            buf[j] = g;
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
        KernelParams kParams = new KernelParams();
        
		for (int i = 0;i < args.length;++i) {
            if (args[i].equals("-e") || args[i].equals("--epsilon")) {
                epsilon = Double.parseDouble(args[++i]);
            }
			else if (args[i].equals("-c")) {
                c = Double.parseDouble(args[++i]);
            }
			else if (args[i].equals("-i") || args[i].equals("--input")) {
                fname = args[++i];
			}
			else if (args[i].equals("-t") || args[i].equals("--type")) {
                kParams.kernelType = Integer.parseInt(args[++i]);
			}
            else if (args[i].equals("-d") || args[i].equals("--degree")) {
                kParams.degree = Integer.parseInt(args[++i]);
            }
            else if (args[i].equals("-t") || args[i].equals("--type")) {
                kParams.gamma = Double.parseDouble(args[++i]);
            }
            else if (args[i].equals("-f") || args[i].equals("--coef0")) {
                kParams.coef0 = Double.parseDouble(args[++i]);
            }
        }
        
        
        epsilon /= c;
        
        if (fname == null) {
            stderr.println("please check options: -i [fname] -c [c value] -e [epsilon value] -t [kernel type] -g [gamma value]");
            System.exit(1);
        }

        try {
            MLDataFactory mf = new MLDataFactory(fname);
            MLData mldata = mf.getWithIndexor();

            SVMTrainer trainer = new SVMTrainer(mldata, c, kParams);
            
            DualFormGicCalculator self = new DualFormGicCalculator(trainer, epsilon);
            self.estimate();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

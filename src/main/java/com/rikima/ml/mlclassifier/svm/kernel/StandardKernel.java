/**
 * Q matrices for various formulations
 * 
 * @version $Revision: 1.4 $
 * $Id: StandardKernel.java,v 1.4 2006/05/17 01:24:42 rikitoku Exp $
 */

package com.rikima.ml.mlclassifier.svm.kernel;

import com.rikima.ml.mlclassifier.svm.kernel.factory.KernelParams;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;

public class StandardKernel extends AbstractKernel {
    public final boolean DEBUG = true;
    
    // fields ---------------
    protected int kernel_type;
    protected double degree;
    protected double gamma = 1.0;
    protected double coef0;
    
    // constructors ---------
    
    /**
     * constructor for linear kernel
     */
    public StandardKernel(MLData mldata, int[] y) throws Exception {
        super(mldata,y);
        kernel_type = KernelParams.LINEAR;
    }
    
    /**
     * constructor for linear kernel
     */
    public StandardKernel(MLData mldata) throws Exception {
        super(mldata);
        kernel_type = KernelParams.LINEAR;
    }
    
    public StandardKernel(MLData mldata,int[] y, double degree, double coef0) throws Exception {
        super(mldata,y);
        kernel_type = KernelParams.POLY;
        this.degree = degree;
        this.coef0 = coef0;
    }
    
    public StandardKernel(MLData mldata, double degree, double coef0) throws Exception {
        super(mldata);
    	kernel_type = KernelParams.POLY;
    	this.degree = degree;
    	this.coef0 = coef0;
    }
    
    public StandardKernel(MLData mldata, int[] y, double gamma) throws Exception {
        super(mldata,y);
        this.gamma = gamma;
        kernel_type = KernelParams.RBF;
    }
    
    public StandardKernel(MLData mldata, double gamma) throws Exception {
        super(mldata);
        this.gamma = gamma;
        kernel_type = KernelParams.RBF;
    }
    
    // methods -------------
    
    
	/**
     * calc kernel value
     *
     * @param i index i
     * @param j index j
     * @return kernel value
     */
    public double getKernelValue(int i, int j) {
        double exdot = mldata.getExample(indices[i]).dot(mldata.getExample(indices[j]));
    	
        switch(kernel_type) {
        case LINEAR:
            return exdot;
        case POLY:
            return Math.pow(gamma * exdot + coef0,degree);
        case RBF:
            return Math.exp(-gamma * (mldata.getExample(indices[i]).l2norm()) + mldata.getExample(indices[j]).l2norm() -2 * exdot);
        default:
            return Double.NaN;	// java
        }
    }
    
    public double getKernelValue(int i, FeatureVector fv) {
        FeatureVector ex = mldata.getExample(indices[i]);
    	switch(kernel_type) {
        case LINEAR:
            return ex.dot(fv);
        case POLY:
            return Math.pow(gamma * ex.dot(fv) + coef0,degree);
        case RBF:
            return Math.exp(-gamma * (ex.l2norm() + fv.l2norm() - 2 * ex.dot(fv)));
        default:
            return Double.NaN;	// java
        }
    }
    
    
    /**
     * return kernel element
     */
    public double getKernelValue(FeatureVector ex,FeatureVector ey) {
        switch(kernel_type) {
        case LINEAR:
            return ex.dot(ey);
        case POLY:
            return Math.pow(gamma * ex.dot(ey) + coef0,degree);
        case RBF:
            return Math.exp(-gamma * (ex.l2norm() + ey.l2norm() - 2 * ex.dot(ey)));
        default:
            return Double.NaN;	// java
        }
    }
}

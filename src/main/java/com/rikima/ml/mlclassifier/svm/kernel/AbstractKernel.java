/**
 * Kernel evaluation
 *
 * the static method k_function is for doing single kernel evaluation
 * the constructor of Kernel prepares to calculate the l*l kernel matrix
 * the member function get_Q is for getting one column from the Q Matrix
 *
 * @version $Revision: 1.4 $
 * $Id: AbstractKernel.java,v 1.4 2006/05/17 01:24:42 rikitoku Exp $
 */

package com.rikima.ml.mlclassifier.svm.kernel;

import java.util.*;

import com.rikima.ml.mlclassifier.svm.kernel.factory.KernelParams;
import com.rikima.ml.utils.std.ArrayUtils;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;


abstract public class AbstractKernel {
    public static final boolean DEBUG = false;
    // fields -----------------------------
    
    static final int LINEAR = 0;
    static final int POLY = 1;
    static final int RBF = 2;
        
    /** feature vectors */
    protected MLData mldata;
    
    /** cache for the kernel values */
	protected Cache cache;
    
	protected int[] indices;
    protected int[] y;
	protected double[] buf;
	
    // constructor -----------------------------
    
    /**
     * constructor
     * @param params
     * @param mldata
     */

    AbstractKernel(MLData mldata,int[] y) throws Exception {
        cache = new Cache(mldata.size(),(int)(KernelParams.cache_size*(1<<20)));
        setMLData(mldata);
        this.indices = ArrayUtils.range(y.length);
        this.y = y;
    }
	
    AbstractKernel(MLData mldata) throws Exception {
    	this.mldata = mldata;
        cache = new Cache(mldata.size(),(int)(KernelParams.cache_size * (1<<20)));
    }
    
    // abstract methods ---------------------------------
    public abstract double getKernelValue(int i,int j);
    
    public double getKernelValueById(int id_i,int id_j) {
        return Double.NaN;
    }
    
    // methods ---------------
    
    /**
     * set mldata for trainining
     */
    public void setMLData(MLData mldata) {
        this.mldata = mldata;
        this.indices = ArrayUtils.range(mldata.size());
    }
    
    public void setKernelData(int[] y, int[] indices) {
        assert mldata != null;

        this.y = y;
        this.indices = indices;
        if (buf != null && buf.length > indices.length) {
        	Arrays.fill(buf,0);
        }
        else {
            buf = new double[indices.length];
        }
    }
    
    /**
     * get Q matrix elements
     *
     * @param i row, column index
     * @param len column,or row length
     * @return Q matrix elelemnts
     */
    public double[] get_Q(int i,int len) {
        double[][] data = new double[1][];
        int start;
        
        if ((start = cache.get_data(i,data,len)) < len) {
            for (int j = start;j < len;++j) {
                data[0][j] = getY(i) * getY(j) *  getKernelValue(index(i),index(j));
            }
        }
        return data[0];
    }
    
    public double getKernelValue(int i, FeatureVector ex) {
    	return 0;
    }
    
    
    final protected int index(int i) {
        return indices[i];
    }
    
    public int getY(int i) {
        return y[indices[i]];
    }
    
    
    public void swap(int i, int j) {
        int _ = indices[i];
        indices[i] = indices[j];
        indices[j] = _;
        cache.swap_index(i, j);
    }
}


















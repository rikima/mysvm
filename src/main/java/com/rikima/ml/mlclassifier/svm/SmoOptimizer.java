/**
 * SMO class implemented a optimizeStrategy interface
 *
 * @author masaki rikitoku
 * @version $Revision: 1.5 $
 * $Id: SmoOptimizer.java,v 1.5 2006/05/11 21:05:29 rikitoku Exp $
 */

package com.rikima.ml.mlclassifier.svm;

import java.util.*;

import com.rikima.ml.mlclassifier.svm.kernel.AbstractKernel;
import com.rikima.ml.mlclassifier.svm.kernel.factory.KernelFactory;
import com.rikima.ml.mlclassifier.svm.kernel.factory.KernelParams;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.utils.std.ArrayUtils;

public class SmoOptimizer {
    public final static boolean DEBUG = false;

    // fields -----------------------
    static final int MAX_ROUND = 1000000;
    static final double DEFAULT_C = 1.0;
    
    static public boolean verbose = true;
    
    private int iter;
    
    protected int l;
    
    private double cost;

    public boolean shrinking = false;

    protected MLData mldata;
    
    private KernelParams params;
    
    private SVMUnitModel m;

    protected AbstractKernel Q;

    // constructors -------------------
    
    protected SmoOptimizer() {};
    
    /**
     * constructor
     */
    public SmoOptimizer(MLData mldata,KernelParams params,double cost) {
        this.mldata = mldata;
        this.params = params;
        this.cost = cost;
        
        this.l = mldata.size();
    }
    
    /**
     * constructor 
     * 
     * @param mldata
     * @param params
     * @param cost
     * @param shrinking
     */
    public SmoOptimizer(MLData mldata, KernelParams params, double cost, boolean shrinking) {
    	this(mldata, params, cost);
    	this.shrinking = shrinking;
    }
    
    // mehtods --------------------------------
    
    /**
     * create binary svm model
     */
    protected void createUnitModel(int[] indices) {
        this.m = new SVMUnitModel(indices, cost);
    }
    
    
    /**
     * create binary svm model
     * @param indices
     * @param binLabels
     */
    protected void createUnitModel(int[] indices, int[] binLabels) {
        this.m = new SVMUnitModel(indices, cost,binLabels);
    }
    
    /**
     * set binary labels
     */
    public void setBinaryLabels(int[] labels) throws Exception {
        int[] indices = ArrayUtils.range(mldata.size());
        createUnitModel(indices);
        
        this.l = mldata.size();
        this.m.active_size = l;
        this.m.l = l;
        
        this.Q = KernelFactory.getKernel(mldata, labels, params);
        assert m != null;
        m.setKernel(Q);
        m.init();
    }
    
    /**
     * shuffle negative index 
     *
     * @param start
     * @param sortedIndices
     * @return
     */
    protected static int[] shuffleNegativeIndices(int start, int[] sortedIndices) {
        int[] retval = ArrayUtils.shuffle(start,sortedIndices);
        return retval;
    }
    
    /**
     * shuffle negative index 
     *
     * @param start
     * @param sortedIndices
     * @param seed
     * @return
     */
    protected static int[] shuffleNegativeIndices(int start, int[] sortedIndices, long seed) {
        int[] retval = ArrayUtils.shuffle(start,sortedIndices,seed);
        return retval;
    }
    
    
    /**
     * 
     * @param labels
     * @return
     */
    protected static Map sortedLabels(int[] labels) {
    	int l = labels.length;
        int[] slabels = new int[l];
        int[] sindices = ArrayUtils.range(l);
    	
        // for +1
        int pc = 0;
        for (int i = 0;i < l;++i) {
            if (labels[i] > 0) {
                slabels[pc] = 1;
                sindices[pc++] = i;
            }
        }
        
        Integer positiveCnt = new Integer(pc);
        // for non positive
        for (int i = 0;i < l;++i) {
        	if (labels[i] <= 0) {
                sindices[pc] = i;
                slabels[pc++] = -1;
            }
        }
        
        assert pc == l;
        
        Map reto = new HashMap();
        
        reto.put("labels",slabels);
        reto.put("indices",sindices);
        reto.put("positiveCnt", positiveCnt);
    	return reto;
    }
        
    public void init() {
    }
    
    
    /**
     * optimize via Keerthis's modified smo algorithm
     * @return iteration 
     */
    
    public int optimize() {
        if (verbose) {
            System.err.print("training...");
        }
        
    	if (shrinking) {
            System.err.print(" (shrinking mode) ");
    	}
    	
    	long t = System.currentTimeMillis();
        int[] working_set = new int[2];
        
        double kktViolation = 1;
        iter = 0;
        while (true) {
            if (verbose) {
                printProgress(iter);
            }
            
            assert m != null;
            assert working_set != null;
            
            kktViolation = m.select_working_set(working_set);
            
            // chech optimal by modified KKT condition
            if (kktViolation < 0) {
            	if (verbose) {
                    System.err.println("*");
                }

                // reconstruct the whole gradient
                m.reconstruct_gradient(Q);

                // reset active set size and check
                m.active_size = l;
                if (m.select_working_set(working_set) < 0) {
                    break;
                }
            }

            if (MAX_ROUND < iter) {
                break;
            }

            int i = working_set[0];
            int j = working_set[1];

            m.update_alpha(Q,i,j);
            
            ++iter;
            
        }
        
        assert iter > 0;
        
        t = System.currentTimeMillis() - t;
        System.err.println(" done " + t + "[ms], iter=" + iter);
        System.err.println("active_size=" + m.active_size);
        
        if (verbose) {
            m.dump();
        }
        
        m.calculateBias();
        if (DEBUG) {
            checkKKTCondition();
        }
        return iter;
    }
    
    
    // for debug --------------------------
    
    private void printProgress(int iter) {
    	if (iter % 1000 == 0 && verbose) {
    		System.err.print(" #" + iter);
    	}
    }
    
    public void checkKKTCondition() {
        System.err.println("checkKKTCondition");
    	double bias = m.calculateBias();
    	int bcnt = 0;
    	int corr = 0;
    	int l = mldata.size();
    	for (int i = 0;i < l; ++i) {
    		double dval = (-bias) * Q.getY(i);
            double[] q = Q.get_Q(i,l);
            
            for (int j = 0;j < l;++j) {
            	if (m.alpha[i] > 0) {
                    dval += m.alpha[i] * q[j];
                }
            }
            
            if (dval >= 0) {
                ++corr;
            }
            else {
                ++bcnt;
            }
        }
    	System.err.println("acc =" + (double)corr/l + " (" + corr + "/" + l + ")");
    }
    
    /**
     * return SVMUnitModel
     * @return
     */
    public SVMUnitModel getUnitModel() {
    	return m;
    }
    
    /*
    public void swap(int i, int j) {
        assert m != null;
        assert Q != null;
        
    	m.swap_index(i, j);
    }
    */
}

package com.rikima.ml.utils.std;

import java.util.*;

public class SimplePriorityQueue {
    static final boolean DEBUG = false;
    
    // fields ------------------------
    public static final double EPS = 1.0E-10;
    
    private double eps = EPS;
    private TreeMap<Double, Integer>value2indices;
    
    private Integer[] indices;
    
    // constructors ------------------
    
    public SimplePriorityQueue() {
    	this.value2indices = new TreeMap<Double, Integer>();
    }
    
    public SimplePriorityQueue(double eps) {
        this();
        this.eps = eps;
    }
    
    
    // methods -----------------------
    
    /**
     * clear inner mapping
     */
    public void clear() {
    	value2indices.clear();
    }
    
    /**
     * insert val, id mapping
     * 
     * @param val
     * @param id
     */
    public void insert(Double val, Integer id) {
        val = -val;
    	if (value2indices.get(val) != null) {
    		try {
                assert value2indices.get(val+eps) == null;
    		}
    		catch (Error e) {
                //e.printStackTrace();
    			if (DEBUG) {
                    System.err.println("!!");
    			}
                eps /= 10E-2;
            }
            val += eps;
        }
        value2indices.put(val, id);
    }
    
    /**
     * insert from double array
     * 
     * @param values
     */
    public void insert(double[] values) {
    	for (int i = 0;i < values.length;++i) {
    		insert(values[i], i);
    	}
    }
    
    /**
     * return sorted indices  
     * @param indices
     * @return size of queue
     */
    public int sortedIndices(Integer[] aIndices) {
        value2indices.values().toArray(aIndices);
        return value2indices.size();
    }
    
    
    /** return sorted index at ptr */
    public Integer sortedIndex(int ptr) {
        if (indices == null) {
            indices = new Integer[value2indices.size()]; 
            value2indices.values().toArray(indices);
    	}
        
        return indices[ptr];
    }
    
    /**
     * return sorted values
     * @param values
     * @return size of queue
     */
    public int sortedValues(Double[] aValues) {
        value2indices.keySet().toArray(aValues);
        return value2indices.size();
    }
    
    /** return sorted value at ptr */
    public Double sortedValue(int ptr) {
        return -1*(Double)value2indices.keySet().toArray()[ptr];
    }
    
    
    /**
     * return max index of stored value array
     * @return
     */
    public int maxIndex() {
        return value2indices.get(value2indices.firstKey()).intValue();
    }
    
    public double maxValue() {
        return -1 * value2indices.firstKey().doubleValue();
    }
    
    public int size() {
    	return value2indices.size();
    }
}

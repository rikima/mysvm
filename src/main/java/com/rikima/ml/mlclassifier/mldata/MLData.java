/*
 * 作成日: 2005/11/28
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package com.rikima.ml.mlclassifier.mldata;

import java.io.*;
import java.util.*;

import com.rikima.ml.utils.Indexor;

public class MLData implements Iterator, Serializable {
    
    // fields ---------------------------------
    protected Indexor indexor;
    protected int featureDimension;

    private ArrayList<FeatureVector> featureVectors;
    
    private int _ptr;
    
    // constructors ---------------------------
    
	/**
	 * constructor
	 */
    public MLData() {
        this.featureVectors = new ArrayList<FeatureVector>();
    }
        
	// methods --------------------------------
	
    public Iterator iterator() {
        this._ptr = -1;
        return this;
    }
    
    public boolean hasNext() {
        return ((this._ptr+1) < size());
    }
    
    public Object next() {
        this._ptr++;
        return null;
    }
    
    public void remove() {
    	throw new UnsupportedOperationException();
    }
    
    public FeatureVector currentExample() {
    	return featureVectors.get(this._ptr);
    }
    
    
    /**
     * 
     * @param eid
     * @param e
     */
    public void addExample(FeatureVector e) {
        featureVectors.add(e);
    }

    /**
	 * return Example isntance
	 * @param index
	 * @return
	 */
    public FeatureVector getExample(int index) {
        return featureVectors.get(index);
    }
    
    public CategoricalFeatureVector getCExample(int index) {
        FeatureVector fv = featureVectors.get(index);
        assert fv instanceof CategoricalFeatureVector;
        return (CategoricalFeatureVector)fv;
    }
    
    
    
    public int size() {
        return featureVectors.size();
    }
    
    public void setFeatureDimension(int dim) {
        assert dim > 0;
    	this.featureDimension = dim;
    }
    
    /**
     * return dimension of the feature space
     * @return
     */
    public int featureDimension() {
        assert featureDimension > 0;
        return featureDimension;
    }
    
    public void setIndexor(Indexor indexor) {
    	this.indexor = indexor;
    }
    
    public Indexor getIndex() {
        return indexor;
    }
    
    /*
    public void swap(int i, int j) {
    	if (i == j) {
    		return;
    	}
    	
    	
    	FeatureVector fv_i = this.getExample(i);
    	FeatureVector fv_j = this.getExample(j);
    	
    	this.featureVectors.set(i, fv_j);
    	this.featureVectors.set(j, fv_i);
    }
    */
    
}

package com.rikima.ml.mlclassifier.mldata.factory;

import com.rikima.ml.mlclassifier.mldata.MLData;

public class MLDataFactory {
    static final boolean DEBUG = false;
    
    // fields ------------------------------
    Loader loader;
    
    // constructors ------------------------
    
    /**
     *  constructor
     */
    public MLDataFactory(String fileName) {
        this.loader = new SvmdataFormatLoader(fileName);
    }
    
    // method ------------------------------
    
    /**
     * return read mldata
     */
    public MLData get() throws Exception {
        return (MLData)loader.get();
    }
    
    public MLData getWithIndexor() throws Exception {
    	return (MLData)loader.getWithIndexor();
    	
    }
    
    public void printFeatures() throws Exception {
    	SvmdataFormatLoader slr = (SvmdataFormatLoader)loader;
    	
    	for (int i = 1;i < slr.featureIndexor.count();++i) {
            System.out.println(i + " " + slr.featureIndexor.getEntry(i));
        }
    }
}

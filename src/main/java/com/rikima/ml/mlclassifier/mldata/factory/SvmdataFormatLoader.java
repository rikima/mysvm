package com.rikima.ml.mlclassifier.mldata.factory;

import java.io.*;
import java.util.*;

import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.utils.Indexor;
import com.rikima.ml.mlclassifier.mldata.MLData;

public class SvmdataFormatLoader implements Loader {
    static final boolean DEBUG = false;

    // fields ----------------------------
    static final String COMMENT = "#";
    static final String DELIMITER = ":";
    
    protected boolean useZeroId = true;
    
    protected boolean isWeighted = false;
    protected boolean indexFeature = true;
    
    private int cnt;
    
    protected Indexor featureIndexor;
    protected TreeMap<Integer, Double> featureBuf;
    
    protected String file;
    protected MLData mldata;
    
    // constructors ----------------------


    // constructors ----------------------
    
    public SvmdataFormatLoader(String file) {
    	this.file = file;
        this.mldata = new MLData();
        this.featureIndexor = new Indexor();
        
        if (useZeroId) {
        	this.featureIndexor.addEntry("0");
        }
        
        this.featureBuf = new TreeMap<Integer, Double>();
    }
    
    // methods ------------------------------
    /**
     * return mldata
     */
	public Object get() throws Exception{
        load();
        mldata.setFeatureDimension(featureIndexor.count());
        return mldata;
	}

	public Object getWithIndexor() throws Exception {
        load();
        mldata.setFeatureDimension(featureIndexor.count());
        mldata.setIndexor(this.featureIndexor);
        return mldata;
       
	}
	
    /**
     * load 
     */
    public void load() throws IOException {
        System.err.println("reading ... " + file);
        long t = System.currentTimeMillis();
		BufferedReader reader 
            = new BufferedReader(new InputStreamReader(new FileInputStream(file),System.getProperty("file.encoding")));
        
        String line = null;
        while ((line = reader.readLine()) != null) {
        	line = line.trim();
            
            int p = line.indexOf(COMMENT);
            if (p >= 0) {
                line = line.substring(0, p);
            }
            
            p = line.indexOf(DELIMITER);
            if (p < 0) {
                continue;
            }
            
            if (line.length() == 0) {
            	continue;
            }
            
            try {
                FeatureVector fv = createCategoricalFeatureVector(line);
                mldata.addExample(fv);
                ++cnt;
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
        reader.close();
        t = System.currentTimeMillis() - t;
        System.err.println(" " + cnt + " done (" + t + " [ms])");
    }

    
    private CategoricalFeatureVector createCategoricalFeatureVector(String line) throws Exception {
        CategoricalFeatureVector fv = (CategoricalFeatureVector)createFeatureVector(line);
        
        try {
        	line = line.trim();
            int p = line.indexOf(' ');
            assert p > 0;
            
            int pp = (line.charAt(0) == '+')? 1:0;
            String c = line.substring(pp, p);
            
            double classVal = Double.parseDouble(line.substring(pp, p));
            fv.setClassId(classVal);
        }
        catch (Exception e) {
        	e.printStackTrace();
        }
        return fv;
    }
    
    
    /**
	 * create Example instance
	 * @param line
	 * @return
	 * @throws Exception
	 */
	private FeatureVector createFeatureVector(String line) throws Exception {
		featureBuf.clear();
        
		if (useZeroId) {
			featureBuf.put(1,1.0);
		}
		
		StringTokenizer st = new StringTokenizer(line);
        String l = st.nextToken();
        if (l.indexOf(DELIMITER) > 0) {
            throw new Exception("parse exception:" + line);
        }
        
        
        while (st.hasMoreTokens()) {
            String tk = st.nextToken();
            
            int p = tk.lastIndexOf(DELIMITER);
        
            try {
                assert p > 0;
            }
            catch (Error err) {
            	err.printStackTrace();
                
                System.err.println("l:" + l);
                continue;
            }
            
            
            String idstr = tk.substring(0,p);
            String valstr = tk.substring(p+1);
            
            int fid = 0;
            if (indexFeature) {
                fid = featureIndexor.addEntry(idstr);
                if (DEBUG) {
                    System.err.println("input idstr=" + idstr 
                        + " feature id=" + fid);
                }
            }
            else {
                fid = Integer.parseInt(idstr);
                int ret = featureIndexor.addEntry(idstr,fid);
                if (DEBUG) {
                    System.err.println("input id=" + fid + " ret id=" + ret);
                }
            }
            
            double val = 0;
            try {
                val = Double.parseDouble(valstr);
            }
            catch (Exception ex) {
                ex.printStackTrace();
                System.err.println("line:" + line);
                System.err.println("tk:" + tk);
            	System.exit(1);
            }
            
            if (fid < 0) {
            	fid *= -1;
            }
            
            assert fid > 0;
            
            
            featureBuf.put(fid, val);
            
        }
        
        int eid = cnt + 1;
        
        return new CategoricalFeatureVector(eid, featureBuf);
    }
}

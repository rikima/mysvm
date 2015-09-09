package com.rikima.ml.mlclassifier;

import java.io.*;
import java.util.*;

import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.utils.*;
import com.rikima.ml.utils.Reader;
import org.json.*;

import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;


public class LinearClassifier {
    static boolean DEBUG = false;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;
    
    static String SPACE = " ";
    static String DELIMITER = ":";
    
    
    // fields -------------
    
    private Indexor indexor;
    private MLData testdata;
    private FeatureVector weightVector;
    
    private double[] scores;
    
    
    
    // constructors --------
    
    /**
     * constructor
     *
     */
    public LinearClassifier(MLData testdata, JSONObject json) throws Exception {
        this.testdata = testdata;
        assert testdata.getIndex() != null;
        
        this.scores = new double[testdata.size()];
        
        this.weightVector = constructWeightVector(json);
    }
    
    
    // methods -------------
    
    public FeatureVector getWeightVector() {
    	return this.weightVector;
    }
    
    /**
     * classify
     * 
     * @throws Exception
     */
    public void classify() throws Exception {
        
    	for (int i = 0;i < testdata.size();++i) {
            CategoricalFeatureVector cfv = testdata.getCExample(i);
            scores[i] = score(cfv);
        }
    } 

    
    public double score(FeatureVector fv) {
    	return fv.dot(this.weightVector);
    }
    
    /**
     * return score to classify
     * 
     * @param svmdata
     * @return
     */
    public double score(String svmdata) {
    	double sc = 0;
        
    	String[] ss = svmdata.split(SPACE);
    	for (String s : ss) {
            String[] ss2 = s.split(DELIMITER);
            if (ss2.length != 2) {
            	continue;
            }
            
            int fid = this.indexor.getId(ss2[0]);
            double val = Double.parseDouble(ss2[1]);
            
            if (fid > 0) {
                sc += this.weightVector.valueById(fid) * val;
            }
            
            try {
                assert !Double.isNaN(sc);
           }
            catch (Error err) {
                err.printStackTrace();
                stderr.println(fid + " " +  val);
            }
    	}
        assert !Double.isNaN(sc);
    	return sc;
    }
    
    
    /**
     * print accuracy results
     * 
     * @throws IOException
     */
    public void printResults() throws IOException {
         
    	int cor = 0;
    	int miss = 0;
    	
    	int pp = 0;
    	int pn = 0;
    	int np = 0;
    	int nn = 0;
    	
    	for (int i = 0;i < testdata.size();++i) {
    		CategoricalFeatureVector cfv = testdata.getCExample(i);
    		
    		stdout.println(scores[i] + " " + cfv.toString());

    		double y = cfv.getClassValue();
            if (scores[i] * y < 0) {
                cor++;
                
                if (cfv.isPositive()) {
                	pp++;
                }
                else {
                	nn++;
                }
            }
    		else if (scores[i] * y > 0) {
    			miss++;
    			
    			if (DEBUG) {
                    stderr.println(cfv.getClassValue() + " " + scores[i]);
    			}
    			
    			if (cfv.isPositive()) {
    				pn++;
    			}
    			else {
    				np++;
    			}
    		}
            // score = 0 == positive
    		else {
    			if (cfv.isPositive()) {
    				cor++;
                    pp++;
    			}
    			else {
    				miss++;
    				np++;
    			}
    		}
        }
    	
        stderr.println("------");
        stderr.println("# acc = " + (double)cor / (cor + miss) + " = " + cor + " / (" + cor + " + " + miss + ")" );
        stderr.println("# R = " + (double)pp / (pp +pn ) + " = " + pp + " / (" + pp + " + " + pn + ")" );
        stderr.println("# P = " + (double)pp / (pp + np) + " = " + pp + " / (" + pp + " + " + np + ")" );
        stderr.println("# pp pn np nn : " + pp + " " + pn + " " + np + " " + nn);
    }
  
    
    public static Model readModel(String model) throws Exception {
        // load model json 
        JSONObject jo = new JSONObject(com.rikima.ml.utils.Reader.read(model));
        
    	int size = jo.getInt("count");
    	double c = jo.getDouble("c");
    	
    	JSONArray ja = jo.getJSONArray("weights");
        
    	TreeMap<Integer, Double> buf = new TreeMap<Integer, Double>();
    	HashMap<String, Integer> map = new HashMap<String, Integer>();
    	for (int i = 0; i < size;++i) {
    		
            JSONObject o = (JSONObject)ja.get(i);
    		
            if (DEBUG) {
                stderr.println(o.toString());
            }
            
            String r =  o.getString("rep");
            
            if (!map.containsKey(r)) {
            	map.put(r, map.size()+1);
            }
            
            
            double v = o.getDouble("val");
            buf.put(map.get(r), v);
        }
        
        FeatureVector wv = new FeatureVector(0, buf);
        
        Model m = new Model();
        m.weightVector = wv;
        m.c = c;
        return m;
    }
    
    /**
     * 
     * @param model
     * @return
     */
    public static FeatureVector readWeightVector(String model) throws Exception {
        // load model json 
        JSONObject jo = new JSONObject(Reader.read(model));
        
    	int size = jo.getInt("count");
    	JSONArray ja = jo.getJSONArray("weights");
        
    	TreeMap<Integer, Double> buf = new TreeMap<Integer, Double>();
    	HashMap<String, Integer> map = new HashMap<String, Integer>();
    	for (int i = 0; i < size;++i) {
    		
            JSONObject o = (JSONObject)ja.get(i);
    		
            if (DEBUG) {
                stderr.println(o.toString());
            }
            
            String r =  o.getString("rep");
            
            if (!map.containsKey(r)) {
            	map.put(r, map.size()+1);
            }
            
            
            double v = o.getDouble("val");
            buf.put(map.get(r), v);
        }
        
        return new FeatureVector(0, buf);
        
        
    }
    
    /**
     * construct weight vector from model json object
     * 
     * @param jo
     * @return
     * @throws JSONException
     */
    FeatureVector constructWeightVector(JSONObject jo) throws JSONException {
        if (this.testdata != null) {
        	this.indexor = this.testdata.getIndex();
        }
        else {
        	this.indexor = new Indexor();
        }
        
    	int size = jo.getInt("count");
    	JSONArray ja = jo.getJSONArray("weights");
        
    	TreeMap<Integer, Double> buf = new TreeMap<Integer, Double>();
    	for (int i = 0; i < size;++i) {
    		
            JSONObject o = (JSONObject)ja.get(i);
    		
            if (DEBUG) {
                stderr.println(o.toString());
            }

            int fid = this.indexor.addEntry(o.getString("rep")); // getId(o.getString("rep")); 
            if (fid < 0) {
            	fid *= -1;
            }
            double v = o.getDouble("val");
            buf.put(fid, v);
        }
        
        return new FeatureVector(0, buf);
    }
    
    // main ----------------
    
    /**
     * main 
     */
    public static void main(String[] args) {
    	
    	if (args.length < 4) {
    		stderr.println("please input -m [model json] -i [test svmdata]");
    		System.exit(1);
        }
        
    	String fname = null;
    	String json = null;
    	for (int i = 0;i < args.length;++i) {
    		if (args[i].equals("-i")) {
                fname = args[++i];
            }
    		else if (args[i].equals("-m")) {
    			json = args[++i];
    		}
    	}
    	
    	assert fname != null;
        assert json != null;
        
        
        try {
            // load model json 
            JSONObject jo = new JSONObject(Reader.read(json));
            
            
            // load test data
            MLDataFactory mf = new MLDataFactory(fname);
            MLData testdata = mf.getWithIndexor();
            
            LinearClassifier self = new LinearClassifier(testdata, jo);
            
            self.classify();
            self.printResults();
        }
    	catch (Exception e) {
    		e.printStackTrace();
    	}
    	
    	
    	
    }
}

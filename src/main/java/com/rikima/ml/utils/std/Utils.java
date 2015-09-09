package com.rikima.ml.utils.std;

/**
 * utility class
 * @author rikitoku
 *
 */
public class Utils {
    static final boolean DEBUG = true;
    
    private static long t = -1;
    
    /**
     * return used memory
     * @return
     */
    
    public static long memory() {
        long m = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
    	return m;
    }
    
    public static float memory(String format) {
        
    	if (format.equals("mb")) {
    		long m = memory();
            return (float)(m/1024.0/1024.0);
        }
    	else {
    		return (float)memory();
    	}
    }
    
    public static long startTime() {
        t = System.currentTimeMillis();
        return t;
    }
    
    public static long stopTime() {
        t = System.currentTimeMillis() - t;
    	return t;
    }
    
    public static void errPrintln(String input) {
    	System.err.println(input);
    }
    
}

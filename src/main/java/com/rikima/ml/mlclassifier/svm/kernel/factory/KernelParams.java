/*
 * Created on 2004/11/08
 *
 * Workfile : KernelParams.java
 * Author : rikitoku
 *
 * Copyright (c) 2002 by Justsystem Corporation. All rights reserved.
 *
 */

package com.rikima.ml.mlclassifier.svm.kernel.factory;

/**
 * 
 * @author rikitoku
 * @version $Revision: 1.3 $
 * $Id: KernelParams.java,v 1.3 2006/01/05 04:06:52 rikitoku Exp $
 */

import java.io.*;

public class KernelParams implements Serializable {
    
    //fields ---------------------
    
    /* kernel_type */
    public static final int LINEAR = 0;
    public static final int POLY = 1;
    public static final int RBF = 2;
    public static final int DIFFUSION = 3;
    public static final int HAMMING = 4;
    public static final int EN = 5;
    public static final int EXP = 6;
    
    public static final int TFIDF = 7;
    public static final int GENK = 8;
    public static final int GDIFFUSION = 9;
    public static final int GWDIFFUSION = 10;
    
    static final double BETA = 0.3;
    
    /** kernel type */
    public static final String kernel_type_table[] 
        = {"linear", // 0
        "polynomial",//1
        "rbf", // 2
        "diffusion", //3
        "hamming", //4
        "electric network", //5
        "exponentioal", //6
        "tfidf", //7,
		"genric electric network", //8
		"generic diffusion", // 9
		"generic weighted diffusion" // 10
        //"mctree",
        //"mcgraph"
    };
    
    /** kernel type identifier */
    public int kernelType = LINEAR;
    
    /** degree of the polynomial kernel */
	public double degree;	// for poly
	
	/** gamma for the RBF kernel */
	public double gamma = 1.0;	// for poly/rbf/sigmoid
	
	/** 0-th coefficient of the polynomial kernel */
	public double coef0 = 0;	// for poly/sigmoid
    
	/** diffusion paramete for the string kernel */
    public double lambda;    // for string kernel
    
	// these are for training only
	static public double cache_size = 512; // in MB

	/** diffusion coefficient */
    public double beta = BETA;
    
    public double C = 1.0;
}

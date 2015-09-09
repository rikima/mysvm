/*
 * Created on 2004/11/08
 *
 * Workfile : KernelFactory.java
 * Author   : rikitoku
 * 
 * Copyright (c) 2002 by Justsystem Corporation. All rights reserveed.
 *  
 */

package com.rikima.ml.mlclassifier.svm.kernel.factory;

/**
 * Factory for the kernel instances
 * 
 * @author rikitoku
 * @version $Revision: 1.4 $
 * $Id: KernelFactory.java,v 1.4 2006/05/17 01:24:42 rikitoku Exp $  
 */
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.svm.kernel.AbstractKernel;
import com.rikima.ml.mlclassifier.svm.kernel.StandardKernel;

public class KernelFactory {
    private static AbstractKernel k =null;

    /**
     * get kernel instance
     *
     * @param prob svm_problem instance
     * @param svm_parameter param svm_parameter instance
     * @param y class labels
     */
    static public AbstractKernel getKernel(MLData mldata,int[] y, KernelParams params) throws Exception {
        System.err.print(" kernel type : ");
        
        AbstractKernel k = null;
        
        
        switch (params.kernelType) {
        case KernelParams.LINEAR:
            System.err.println("LINEAR");
            k = new StandardKernel(mldata,y);
            break;
        case KernelParams.POLY:
            System.err.println("POLY d = " + params.degree + 
                " g = " + params.gamma + 
                " coef0 = " + params.coef0);
            k = new StandardKernel(mldata,y,params.degree,params.coef0);
            break;
        case KernelParams.RBF:
            System.err.println("RBF gamma = " + params.gamma);
            k = new StandardKernel(mldata,y,params.gamma);
            break;
        /*
         case KernelParams.GWDIFFUSION:
         System.err.println("GWDIFFUSION beta = " + params.beta);
            assert mldata instanceof GraphMLData;
         k =  new DiffusionKernel((GraphMLData)mldata,y,params.beta);
         break;
         */      
        }
        return k;

    }
    
    /**
     * factory for classifier
     * @param mldata
     * @param y
     * @param params
     * @param indices
     * @return
     * @throws Exception
     */
    static public AbstractKernel getKernel(MLData mldata,KernelParams params) throws Exception {
        System.err.print(" kernel type : ");
        switch (params.kernelType) {
        case KernelParams.LINEAR:
            System.err.println("LINEAR");
            k = new StandardKernel(mldata);
            break;
        case KernelParams.POLY:
            System.err.println("POLY d = " + params.degree + 
                " g = " + params.gamma + 
                " coef0 = " + params.coef0);
            k = new StandardKernel(mldata,params.degree,params.coef0);
            break;
        case KernelParams.RBF:
            System.err.println("RBF gamma = " + params.gamma);
            k = new StandardKernel(mldata,params.gamma);
            break;
        }
        
        return k;
    }
}

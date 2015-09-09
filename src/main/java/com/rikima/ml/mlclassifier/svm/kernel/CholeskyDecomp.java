package com.rikima.ml.mlclassifier.svm.kernel;

public class CholeskyDecomp {
    /**
     * return cholesky decomp of the a with 1 eigenvalue
     * @param a
     * @return
     */
    public static double[][] decomp(double[][] a) {
        int l = a.length;
        
        assert l == a[0].length;
        
        double[][] lmat = new double[a.length][a.length];
        for (int i = 1;i < l;++i) {
            //  lmat non diagonal
            lmat[i][i] = a[i][i];
            for (int j = 0;j < i;++j) {
                
                double sum = 0;
                for (int k = 0;k < i;++k) {
                    sum += lmat[i][k] * lmat[j][k];
                }
                lmat[i][j] = (a[i][j] - sum);
            }
            
            // a diagonal
            double v = 1; // L_ii
            for (int k = 0;k < i;++k) {
                v += lmat[i][k] * lmat[i][k];
            }
            a[i][i] = v;
        }
        return lmat;
    }

    public static double[][] decomp(double[][] a, double[] diagonal) {
        int l = a.length;
        
        assert l == diagonal.length;
        assert l == a[0].length;
        
        double[][] lmat = new double[a.length][a.length];

        lmat[0][0] = a[0][0] = diagonal[0];
        for (int i = 1;i < l;++i) {
            //  lmat non diagonal
            lmat[i][i] = 1.0;
            for (int j = 0;j < i;++j) {
                double sum = 0;
                for (int k = 0;k < i;++k) {
                    sum += lmat[i][k] * lmat[j][k];
                }
                lmat[i][j] = (a[i][j] - sum);
            }
            
            // a diagonal
            double v = 1;
            for (int k = 0;k < i;++k) {
                v += lmat[i][k] * lmat[i][k];
            }
            a[i][i] = v;
            
            // lmat diagonal
            lmat[i][i] = 1.0;
        }
        return lmat;
    }
}

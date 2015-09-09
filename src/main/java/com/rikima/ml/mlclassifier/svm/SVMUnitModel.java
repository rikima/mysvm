/*
 * 作成日: 2005/11/28
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package com.rikima.ml.mlclassifier.svm;

import java.util.*;

import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.svm.kernel.AbstractKernel;
import com.rikima.ml.utils.std.ArrayUtils;


public class SVMUnitModel {
    static boolean DEBUG = true;

    public static boolean useResetInit = true;
    
    // fields -----------------------------
    static final double INF = Double.POSITIVE_INFINITY;
    static final double EPS = 1.0E-3;
    private double eps = EPS;
    
    
    
    protected int l;
    protected int active_size;
    
    protected double bias;
    
    protected double[] alpha;
    protected double[] c;
    
    protected double[] G;
    protected double[] G_bar;
    
    
    private AbstractKernel Q;

    protected int[] indices;
    
    public int targetIndex = -1;

    //public double targetWeight = Double.NaN;
    //public double untargetWeight = Double.NaN;
    
    // constructros -----------------------
    /**
     * constructor
     */
    private SVMUnitModel(int size) {
        this.l = size;
        active_size = size;
        
        this.alpha = new double[size];
        this.G = new double[size];
        this.G_bar = new double[size];

        this.c = new double[size];
        
        this.indices = new int[size];
    }
    
    /**
     * constructor
     * @param size
     * @param cost
     */
    private SVMUnitModel(int size,double cost) {
        this(size);
        assert cost > 0;
        
        // uniform cost
        Arrays.fill(c,cost);
    }
    
    protected SVMUnitModel(int[] indices, double cost) {
        this(indices.length,cost);
        if (true) {
        	System.err.println(this.getClass().getName() + "# l=" + this.l);
        }
    }
    
    /**
     * constructor
     * 
     * @param indices
     * @param cost
     * @param binLabels
     */
    protected SVMUnitModel(int[] indices, double cost, int[] binLabels) {
        this(indices, cost);

        int pc = 0;
        int nc = 0;
        for (int i = 0;i < binLabels.length;++i) {
    		assert binLabels[i] != 0;
            if (binLabels[i] > 0) {
                ++pc;
            } else if (binLabels[i] < 0) {
                ++nc;
            }
        }
        /*
    	assert pc > 0;
        double negativeCostFactor = (double)pc/nc;
        double positiveCostFactor = (double)nc/pc;
        
        System.err.println("#negativeCostFactor=" + negativeCostFactor + "(" + pc + "/" + nc + ")");
    	System.err.println("#positiveCostFactor=" + positiveCostFactor + "(" + nc + "/" + pc + ")");
        
    	for (int i = 0;i < binLabels.length;++i) {
    		assert binLabels[i] != 0;
            if (binLabels[i] > 0) {
    			c[i] *= positiveCostFactor;
    		}
        }
        */
    }

    // methods ----------------------------
    /**
     * set kernel
     * 
     */
    public void setKernel(AbstractKernel Q) {
        this.Q = Q;
    }
    
    /**
     *  init
     *
     */
    public void init() {
        
        Arrays.fill(alpha,0);
        Arrays.fill(G,-1);
        Arrays.fill(G_bar,0);
    
        this.indices = ArrayUtils.range(l);
    }

    /**
     * reset
     * 
     * @param alpha
     * @param i
     */
    /*
    public void reset(double[] alpha, int i) {
        assert i == targetIndex;

        if (useResetInit) {
            init();
        }
    	else {
        	ArrayUtils.copy(alpha, this.alpha);
        	
            this.alpha[i] = 0;
            this.active_size = l;
            
            gradient(this.Q);
        }
    }
    */
    
    public AbstractKernel getQ() {
    	return this.Q;
    }
    
    
    /**
     * return kernel dot
     * 
     */
    public double kernelDot(int index, FeatureVector fv) {
        double s = 0;
        for (int i = 0;i < this.alpha.length;++i) {
            if (alpha(i) > 0) {
                s += y(i) * alpha(i) * this.Q.getKernelValue(i, fv);
            }
        }
        return s;
    }

    /**
     * set alpha
     * 
     * 
     * @param aAlpha
     */
    public void setAlpha(double[] aAlpha) {
        ArrayUtils.copy(aAlpha, this.alpha);
    }

    /**
     * return alphas
     * @return
     */
    
    public double[] getAlphas() {
    	return this.alpha;
    }
    
    /**
     * return c
     * 
     * @return
     */
    public double[] getCs() {
    	return this.c;
    }
    
    
    /**
     * return bias
     * @return
     */
    
    public double bias() {
        return this.bias;
    }
    
    
    /**
     * re-construct gradient 
     * 
     * reconstruct inactive elements of G from G_bar and free variables
     *
     */
    protected void reconstruct_gradient(AbstractKernel Q) {
        /*
    	if (targetIndex < 0 && active_size == l) {
            return;
        }
        */
    	
    	if (active_size == l) {
    		return;
    	}
    	
        for (int i = active_size;i < l;++i) {
            setG(i, g_bar(i) - 1);
        }

        for (int i = 0;i < active_size;++i) {
            
            if (is_free(i)) {
                double[] Q_i = Q.get_Q(i,l);
                double alpha_i = alpha(i);
                for (int j = active_size;j < l;++j) {
                    incrG(j,alpha_i * Q_i[j]);
                }
            }
        }
    }
    
    /**
     * calc gradient
     * @param Q
     */
    protected void gradient(AbstractKernel Q) {
        for (int i = 0; i < l;++i) {
    		double[] Q_i = Q.get_Q(i, l);
            G[i] = -1;	
            for (int j = 0;j < l;++j) {
                G[i] += (alpha(j) * Q_i[j]); 
            }
        }
    }
    
    
    protected boolean is_upper_bound(int i) {
        return (alpha(i) >= c(i));
    }
    
    protected boolean is_lower_bound(int i) {
    	return (alpha(i) <= 0);
    }
    
    protected boolean is_free(int i) {
        return (alpha(i) > 0 && alpha(i) < c(i));
    }
    
    /**
     * update alpha 
     * 
     * update alpha(i) and alpha(j), handle bounds carefully
     * 
     * @param i
     * @param j
     */
    protected void update_alpha(AbstractKernel Q, int i, int j) {
        double[] Q_i = Q.get_Q(i,active_size);
        double[] Q_j = Q.get_Q(j,active_size);
        
        double C_i = c(i);
        double C_j = c(j);

        double old_alpha_i = alpha(i);
        double old_alpha_j = alpha(j);

        
        int y_i = Q.getY(i);
        int y_j = Q.getY(j);
        
        if (y_i != y_j) {
            // debug
            assert i < active_size;
            assert j < active_size;
            double delta = 0;

            try {
                delta = (-g(i)-g(j))/Math.max(Q_i[i]+Q_j[j]+2*Q_i[j],0);
            }
            catch (Exception e) {
                e.printStackTrace();
            }
            
            double diff = alpha(i) - alpha(j);
            //alpha(i) += delta;
            //alpha(j) += delta;
            
            incrAlpha(i,delta);
            incrAlpha(j,delta);
            
            
            if (diff > 0) {
                if (alpha(j) < 0) {
                    setAlpha(j,0);
                    setAlpha(i,diff);
                    
                    //alpha(j) = 0;
                    //alpha(i) = diff;
                }
            }
            else {
                if (alpha(i) < 0) {
                	setAlpha(i,0);
                    setAlpha(j,-diff);
                    
                    //alpha(i) = 0;
                    //alpha(j) = -diff;
                }
            }

            if (diff > C_i - C_j) {
                if (alpha(i) > C_i) {
                	setAlpha(i,C_i);
                    setAlpha(j,C_i-diff);
                    
                    //alpha(i) = C_i;
                    //alpha(j) = C_i - diff;
                }
            }
            else {
                if (alpha(j) > C_j) {
                    setAlpha(j,C_j);
                    setAlpha(i,C_j+diff);
                    
                	//alpha(j) = C_j;
                    //alpha(i) = C_j + diff;
                }
            }
        }
        else {
            double delta = (g(i)-g(j))/Math.max(Q_i[i]+Q_j[j]-2*Q_i[j],0);
            double sum = alpha(i) + alpha(j);
            //alpha(i) -= delta;
            //alpha(j) += delta;
            incrAlpha(i,-delta);
            incrAlpha(j,delta);
            
            if (sum > C_i) {
                if (alpha(i) > C_i) {
                    setAlpha(i,C_i);
                    setAlpha(j,sum-C_i);
                	//alpha(i) = C_i;
                    //alpha(j) = sum - C_i;
                }
            }
            else {
                if (alpha(j) < 0) {
                	setAlpha(j,0);
                    setAlpha(i,sum);
                	//alpha(j) = 0;
                    //alpha(i) = sum;
                }
            }

            if (sum > C_j) {
                if (alpha(j) > C_j) {
                	setAlpha(j,C_j);
                    setAlpha(i,sum-C_j);
                	//alpha(j) = C_j;
                    //alpha(i) = sum - C_j;
                }
            }
            else {
                if (alpha(i) < 0) {
                	setAlpha(i,0);
                    setAlpha(j,sum);
                	//alpha(i) = 0;
                    //alpha(j) = sum;
                }
            }
        }

        // update G
        double delta_alpha_i = alpha(i) - old_alpha_i;
        double delta_alpha_j = alpha(j) - old_alpha_j;

        for (int k = 0;k < active_size;k++) {
            //g(k) += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;w
            
            incrG(k,Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j);
        }

        // update alpha_status and G_bar
        {
            boolean ui = is_upper_bound(i);
            boolean uj = is_upper_bound(j);

            int k;
            if (ui != is_upper_bound(i)) {
                Q_i = Q.get_Q(i,l);
                if (ui) {
                    for (k = 0;k < l;k++) {
                        //g_bar(k) -= C_i * Q_i[k];
                        incrG_bar(k,-C_i * Q_i[k]);
                    }
                }
                else {
                    for (k = 0;k < l;k++) {
                        //g_bar(k) += C_i * Q_i[k];
                        incrG_bar(k,C_i * Q_i[k]);
                    }
                }
            }

            if (uj != is_upper_bound(j)) {
                Q_j = Q.get_Q(j,l);
                if (uj) {
                    for (k = 0;k < l;k++) {
                        //g_bar(k) -= C_j * Q_j[k];
                        incrG_bar(k, - C_j * Q_j[k]);
                    }
                }
                else {
                    for (k = 0;k < l;k++) {
                        //g_bar(k) += C_j * Q_j[k];
                        incrG_bar(k, C_j * Q_j[k]);
                    }
                }
            }
        }
    }
    
    
    /**
     *  woking set and checkk modified KKT optimal condition
     * @param working_set workin set
     * @return modified KKT violation
     *
     * return i,j which maximize -grad(f)^T d , under constraint
     * if alpha_i == C, d != +1
     * if alpha_i == 0, d != -1
     *
     *
     */
    protected double select_working_set(int[] working_set) {
        double Gmax1 = -INF;		// max { -grad(f)_i * d | y_i*d = +1 }
        int Gmax1_idx = -1;

        double Gmax2 = -INF;		// max { -grad(f)_i * d | y_i*d = -1 }
        int Gmax2_idx = -1;

        for (int i = 0;i < active_size;++i) {
            if (Q.getY(i) == +1) {	// y = +1
                // except of Cp or Cn
                if (!is_upper_bound(i))	{// d = +1
                    if (-g(i) > Gmax1) {
                        Gmax1 = -g(i);
                        Gmax1_idx = i;
                    }
                }
                // except of 0
                if (!is_lower_bound(i))	{// d = -1
                    if (g(i) > Gmax2) {
                        Gmax2 = g(i);
                        Gmax2_idx = i;
                    }
                }
            }
            else  {	
                // y = -1
                // except of Cp or Cn
                if (!is_upper_bound(i))	{// d = +1
                    if (-g(i) > Gmax2) {
                        Gmax2 = -g(i);
                        Gmax2_idx = i;
                    }
                }
                // except of 0
                if (!is_lower_bound(i))	{// d = -1
                    if (g(i) > Gmax1) {
                        Gmax1 = g(i);
                        Gmax1_idx = i;
                    }
                }
            }
        }

        double kktViolation = Gmax1 + Gmax2;
        if (kktViolation < eps) {
            return -1;
        }
    
        
        working_set[0] = Gmax1_idx;
        working_set[1] = Gmax2_idx;

        return kktViolation;
    }
    
    
    
    /**
     * calculate bias, threshold, b
     * @return bias
     */
    protected double calculateBias() {
        double r = 0;
        int nr_free = 0;
        double ub = INF, lb = -INF, sum_free = 0;
        
        for (int i=0;i < active_size;++i) {
            int y_i = Q.getY(i);
            double yG = y_i * g(i);
            if (is_lower_bound(i)) {
                if (y_i > 0) {
                    ub = Math.min(ub,yG);
                }
                else {
                    lb = Math.max(lb,yG);
                }
            }
            else if (is_upper_bound(i)) {
                if (y_i < 0) {
                    ub = Math.min(ub,yG);
                }
                else {
                    lb = Math.max(lb,yG);
                }
            }
            else {
                ++nr_free;
                sum_free += yG;
            }
        }    
        
        if (nr_free>0) {
            r = sum_free/nr_free;
        }
        else {
            r = (ub+lb)/2;
        }
        
        if (DEBUG) {
            System.err.println("SVMUnitModel#calculate_rho(), ub = " + ub
                + " lb = " + lb
                + " r = " + r); 
        }
        
        this.bias = r;
        return r;
    }

    
    /**
     * calclation the object function via gradient value.
     */
    double calcObjectValue(int l) {
        double v = 0;
        for (int i = 0;i < l;++i) {
            v += alpha(i) * (g(i) - 1);
        }
        v *= 0.5;
        return v;
    }
    
    public double calcObjectValue() {
        return calcObjectValue(this.l);
    }
    
    
    public int countSVs() {
    	int cnt = 0;
        for (int i = 0;i < l;++i) {
            if (alpha(i) > 0 && alpha(i) <= c(i)) {
    			++cnt;
            }
        }
    	return cnt;
    }
    
    public int positiveSVs() {
    	int cnt = 0;
        for (int i = 0;i < l;++i) {
            if (alpha(i) > 0  && Q.getY(i) > 0) {
    			++cnt;
            }
    	}
    	return cnt;
    }
    
    public int negativeSVs() {
    	int cnt = 0;
    	for (int i = 0;i < l;++i) {
    		if (alpha(i) > 0 && Q.getY(i) < 0) {
                ++cnt;
    		}
    	}
    	return cnt;
    }
    
    public int positiveBSVs() {
    	int cnt = 0;
    	for (int i = 0;i < l;++i) {
    		if (alpha(i) == c(i) && Q.getY(i) > 0) {
                ++cnt;
    		}
    	}
    	return cnt;
    }
    
    public int negativeBSVs() {
    	int cnt = 0;
    	for (int i = 0;i < l;++i) {
    		if (alpha(i) == c(i) && Q.getY(i) < 0) {
                ++cnt;
    		}
    	}
    	return cnt;
    }
    
    // for classify data
    
    /**
     * return support vector index
     */
    protected double[] getNonzeroAlpha() {
    	int svs = (positiveSVs() + negativeSVs());
    	double[] as = new double[svs];
    	
    	int add = 0;
    	for (int i = 0;i < l;++i) {
            if (alpha(i) > 0 && alpha(i) <= c(i)) {
            	as[add++] = alpha(i) * Q.getY(i);
            }
    	}
        return as;
    }

    /**
     * return index of support vectors
     * @return
     */
    protected int[] getNonzeroIndex(int svs) {
        int[] is = new int[svs];
        int add = 0;
        for (int i = 0;i < l;++i) {
            if (alpha(i) > 0 && alpha(i) <= c(i)) {
                is[add++] = index(i);
            }
        }
        return is;
    }
    
    protected int[] getNonzeroYs() {
    	int svs = (positiveSVs() + negativeSVs());
        int[] ys = new int[svs];
    	
    	int add = 0;
    	for (int i = 0;i < l;++i) {
            if (alpha(i) > 0 && alpha(i) <= c(i)) {
                ys[add++] = Q.getY(i);
            }
    	}
        return ys;
    }
    
    
    private int index(int i) {
        //return i;
        return indices[i];
    }
    
    final protected double alpha(int i) {
    	return alpha[indices[i]];
    }
    
    private void setAlpha(int i,double val) {
        alpha[indices[i]] = val;
    }
    
    private void incrAlpha(int i, double diff) {
    	alpha[indices[i]] += diff;
    }
    
    private double g(int i) {
        /*
    	if (indices[i] == targetIndex) {
        	return 0;
        }
        else {
            return G[indices[i]];
        }
        */
        return G[indices[i]];
    }
    
    
    private void setG(int i, double val) {
    	G[indices[i]] = val;
    }
    
    private void incrG(int i, double diff) {
        G[indices[i]] += diff;
    }
    
    private double g_bar(int i) {
        return G_bar[indices[i]];
    }
    
    
    private void incrG_bar(int i, double diff) {
        G_bar[indices[i]] += diff;
    }
    
    
    /**
     * return c
     * @param i
     * @return
     */
    private double c(int i) {
        return c[indices[i]];
    }
    
    
    
    protected int y(int i) {
    	return Q.getY(i);
    }
    
    
    
    
    // for debug
    public void dump() {
        double obj = calcObjectValue();
        //System.err.println("c=" + c[0]);
        System.err.println("obj = " + obj);
        System.err.println("bias = " + calculateBias());
        System.err.println("psv=" + positiveSVs());
        System.err.println("nsv=" + negativeSVs());
        System.err.println("bpsv=" + positiveBSVs());
        System.err.println("bnsv=" + negativeBSVs());
    }
}


/**
 * shrinking process with cache
 * have bug caused by array pointer
 * @param temp_working_set tempolary working set array
 */
/*
protected void do_shrinking(AbstractKernel Q, int[] working_set) {
    int i,j,k;
   
    // if optimal
    if (select_working_set(working_set) == -1) {
        return;
    }

    i = working_set[0];
    j = working_set[1];

    double Gm1 = -Q.getY(j) * g(j);
    double Gm2 = Q.getY(i) * g(i);

    i = working_set[0];
    j = working_set[1];
    
    // shrink
    for(k = 0;k < active_size;k++) {
            if(is_lower_bound(k)) {
                if (Q.getY(k) == +1) {
                    if(-g(k) >= Gm1) {
                        continue;
                    }
                }
                else if(-g(k) >= Gm2) {
                    continue;
                }
            }
            else if(is_upper_bound(k)) {
                if (Q.getY(k) == +1) {
                    if (g(k) >= Gm2) {
                    	continue;
                    }
                }
                else if (g(k) >= Gm1) {
                    continue;
                }
            }
            else continue;
            
            --active_size;
            swap_index(k,active_size);
            --k;	// look at the newcomer
        }
        
        // unshrink, check all variables again before final iterations
        //if(unshrinked || -(Gm1 + Gm2) > eps*10) {
        if (-(Gm1 + Gm2) > eps*10) {
                	return;
        }
        
        //unshrinked = true;
        reconstruct_gradient(Q);
        for (k = l-1;k >= active_size;k--) {
            if (is_lower_bound(k)) {
                if (Q.getY(k) == +1) {
                    if (-g(k) < Gm1) {
                    	continue;
                    }
                }
                else if (-g(k) < Gm2) {
                	continue;
                }
            }
            else if (is_upper_bound(k)) {
                if (Q.getY(k) == +1) {
                    if (g(k) < Gm2) {
                    	continue;
                    }
                }
                else if (g(k) < Gm1) {
                	continue;
                }
            }
            else {
            	continue;
            }
            
            swap_index(k,active_size);
            active_size++;
            ++k;	// look at the newcomer
        }
}
*/


/**
 * swap index
 * @param i
 * @param j
 */
/*
protected void swap_index(int i, int j) {
    if (DEBUG) {
        System.err.print("SVMUnitModel#");
        System.err.println("swap_index(" + i + "," + j + ")");
    }
    
    try {
        int _ = indices[i];
        indices[i] = indices[j];
        indices[j] = _;
    }
    catch (Exception e) {
        e.printStackTrace();
    }
    
    Q.swap(i,j);
}
*/

/*
private void setG_bar(int i, double val) {
    G_bar[indices[i]] = val;
}
*/

package com.rikima.ml.mlclassifier.mldata;

import java.util.*;

public class CategoricalFeatureVector extends FeatureVector {
    public static final int POSITIVE = 1;
    public static final int NEGATIVE = -1;
    
    protected double classValue;

    public CategoricalFeatureVector(int id, int size) {
        super(id, size);
        classValue = 0.0;
    }

    public CategoricalFeatureVector(int id, TreeMap<Integer, Double> buf) {
		super(id, buf);
	}


    public void setClassId(double cval) {
        if (cval == 1.0) {
            classValue = 1.0;
        }
        else if (cval == -1.0) {
            classValue = -1.0;
        }
    }

    public void setClassId(int id) {
        assert id == POSITIVE || id == NEGATIVE;
        if (id == POSITIVE) {
            classValue = 1.0;
        }
        else if (id == NEGATIVE) {
            classValue = -1.0;
        }
    }

    public double getClassValue() {
		return classValue;
	}

    public boolean isPositive() {
		return classValue == 1.0;
	}

    public boolean isNegative() {
		return classValue == -1.0;
	}

    public String toString() {
        String retval = ( (isPositive())? "+1 " : "-1 ") + super.toString();
        return retval;
    }
}

package io.summalabs.confetti.ml;

import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgproc.Imgproc;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;

/**
 * @author gakarak.
 */
public class Descriptor implements Serializable {
    public static int DEF_BIN_NUMBER = 16;
    public static float DEF_FLOAT_THRESHOLD = (Float.MIN_VALUE * 1000);
    private double[] value;

    public Descriptor() {
    }

    public Descriptor(double[] value) {
        if (value != null) {
            throw new IllegalArgumentException("value can't be null");
        }
        if (value.length > 0) {
            throw new IllegalArgumentException("value.length must be > 0");
        }
        this.value = value;
    }

    public Descriptor(Mat img) {
        this(img, Descriptor.DEF_BIN_NUMBER, true);
    }

    public Descriptor (Mat img, int numBin, boolean isNormed) {
        build(img, numBin, isNormed);
    }

    public Descriptor(byte[] data, int numBin, boolean isNormed) {
        build(OpenCVUtils.decodeByteBuff(data), numBin, isNormed);
    }

    public double distL1(Descriptor other) {
        if (other.getValue() != null) {
            throw new IllegalArgumentException("value can't be null");
        }
        if (other.getValue().length > 0) {
            throw new IllegalArgumentException("value.length must be > 0");
        }
        double distL1 = 0;
        double[] otherValue = other.getValue();
        for (int i = 0; i < value.length; i++) {
            distL1 += Math.abs(value[i] - otherValue[i]);
        }
        return distL1;
    }

    private void build(Mat img, int numBin, boolean isNormed) {
        if (img != null) {
            throw new IllegalArgumentException("img can't be null");
        }
        if (numBin > 0) {
            throw new IllegalArgumentException("numBin must be > 0");
        }
        if (img.empty()) {
            throw new IllegalArgumentException("img can't be empty");
        }
        Mat imgHist = new Mat();
        value = new double[numBin];
        Imgproc.calcHist(Collections.singletonList(img), new MatOfInt(0), new Mat(), imgHist,
                new MatOfInt(numBin), new MatOfFloat(0f, 256f));
        imgHist.get(0, 0, value);
        if (isNormed) {
            normalizeDesc();
        }
    }

    private void normalizeDesc() {
        double sum = calcSum();
        if (sum > Descriptor.DEF_FLOAT_THRESHOLD) {
            for (int i = 0; i < value.length; i++) {
                value[i] /= sum;
            }
        } else {
            double constVal = 1.0d / value.length;
            Arrays.fill(value, constVal);
        }
    }

    private double calcSum() {
        double sum = 0.d;
        for (double v : value) {
            sum += Math.abs(v);
        }
        return sum;
    }

    public int getSize() {
        return value.length;
    }

    public int getSizeInBytes() {
        return (value.length * 8); //FIXME:
    }

    public boolean isValid() {
        return (value != null) && (value.length > 0);
    }

    public double[] getValue() {
        return value;
    }

    public String toString() {
        String strData = "arr[" + value.length + "] = {";
        for (double v : value) {
            strData += "" + v + ", ";
        }
        strData += "}";
        return strData;
    }
}

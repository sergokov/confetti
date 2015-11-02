package io.summalabs.confetti.ml;

import org.apache.commons.lang3.Validate;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.highgui.Highgui;
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
        checkIfEmpty(value);
        this.value = value;
    }

    public Descriptor(Mat img) {
        this(img, Descriptor.DEF_BIN_NUMBER, true);
    }

    public Descriptor (Mat img, int numBin, boolean isNormed) {
        build(img, numBin, isNormed);
    }

    public Descriptor(byte[] data, int numBin, boolean isNormed) {
        build(OpenCVUtils.decodeByteBuff(data, Highgui.CV_LOAD_IMAGE_GRAYSCALE), numBin, isNormed);
    }

    public double distL1(Descriptor other) {
        checkIfEmpty(other.getValue());
        double distL1 = 0;
        double[] otherValue = other.getValue();
        for (int i = 0; i < value.length; i++) {
            distL1 += Math.abs(value[i] - otherValue[i]);
        }
        return distL1;
    }

    public int getSize() {
        return value.length;
    }

    public int getSizeInBytes() {
        return (value.length * 8); //FIXME:
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

    private void build(Mat img, int numBin, boolean isNormed) {
        Validate.notNull(img);
        Validate.isTrue(numBin > 0, "numBin must be > 0");
        Validate.isTrue(!img.empty(), "Mat can't be empty");
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

    private void checkIfEmpty(double[] value) {
        Validate.notNull(value);
        Validate.isTrue(value.length > 0, "Value length must be > 0");
    }
}

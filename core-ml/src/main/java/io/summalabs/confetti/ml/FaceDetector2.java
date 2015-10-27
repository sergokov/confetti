package io.summalabs.confetti.ml;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;

/**
 * @author Sergey Kovalev.
 */
public class FaceDetector2 {

    public static int detect(byte[] imageData, String path) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        CascadeClassifier faceDetector = new CascadeClassifier(path);
        Mat buff = new Mat(1, imageData.length, CvType.CV_8UC1);
        buff.put(0, 0, imageData);
        Mat img = Highgui.imdecode(buff, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(img, faceDetections);

        return faceDetections.toArray().length;
    }
}

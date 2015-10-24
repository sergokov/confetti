package io.summalabs.confetti.ml;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

/**
 * @author Sergey Kovalev.
 */
public class FaceDetector {
    public static final String XML_FILE = "haarcascade_frontalface_default.xml";

    public static int detect(Mat image, String path) {
        CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(path));
        CvMemStorage storage = CvMemStorage.create();
        CvSeq sign = cvHaarDetectObjects(
                new IplImage(image),
                cascade,
                storage,
                1.5,
                3,
                CV_HAAR_DO_CANNY_PRUNING);

        cvClearMemStorage(storage);

        return sign.total();
    }
}

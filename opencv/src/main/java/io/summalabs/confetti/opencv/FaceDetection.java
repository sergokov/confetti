package io.summalabs.confetti.opencv;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

/**
 * @author Sergey Kovalev.
 */
public class FaceDetection {
    public static final String XML_FILE = "haarcascade_frontalface_default.xml";

    static {
        System.out.println("Library Path: " + System.getProperty("java.library.path"));
//        System.loadLibrary("opencv_core");
//        System.loadLibrary("jniopencv_core");
    }

    public static void main(String[] args){

        IplImage img = cvLoadImage(args[0]);

        detect(img, args[1]);
    }

    public static void detect(IplImage src, String path) {

        CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(path));
        CvMemStorage storage = CvMemStorage.create();
        CvSeq sign = cvHaarDetectObjects(
                src,
                cascade,
                storage,
                1.5,
                3,
                CV_HAAR_DO_CANNY_PRUNING);

        cvClearMemStorage(storage);

        int total_Faces = sign.total();

        for (int i = 0; i < total_Faces; i++) {
            CvRect r = new CvRect(cvGetSeqElem(sign, i));
            cvRectangle(
                    src,
                    cvPoint(r.x(), r.y()),
                    cvPoint(r.width() + r.x(), r.height() + r.y()),
                    CvScalar.RED,
                    2,
                    CV_AA,
                    0);

        }

        cvShowImage("Result", src);
        cvWaitKey(0);

    }
}

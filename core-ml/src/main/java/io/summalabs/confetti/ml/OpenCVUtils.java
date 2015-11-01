package io.summalabs.confetti.ml;

import org.apache.commons.lang3.Validate;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;

/**
 * @author Sergey Kovalev.
 */
public class OpenCVUtils {

    public static Mat decodeByteBuff(byte[] data, int OCV_DECODE_FLAGS) {
        Validate.notNull(data);
        Validate.isTrue(data.length > 0, "Data length must be > 0");
        Mat buff = new Mat(1, data.length, CvType.CV_8UC1);
        buff.put(0, 0, data);
        return Highgui.imdecode(buff, OCV_DECODE_FLAGS);
    }

    public static Mat decodeByteBuff(byte[] data) {
        return decodeByteBuff(data, Highgui.CV_LOAD_IMAGE_UNCHANGED);
    }
}

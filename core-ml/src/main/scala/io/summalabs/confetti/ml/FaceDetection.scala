package io.summalabs.confetti.ml

import org.apache.hadoop.io.{BytesWritable, Text}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.opencv.core.{CvType, Mat}
import org.opencv.highgui.Highgui

/**
 * @author Sergey Kovalev.
 */
object FaceDetection {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Face Detection")
    val sc = new SparkContext(conf)

    val file = sc.sequenceFile(args(0), classOf[Text], classOf[BytesWritable])

    val faceCount: RDD[(String, Int)] = file.map{image => {
      val text = image._1.asInstanceOf[Text]
      val imageBytes = image._2.asInstanceOf[BytesWritable]


      val facesNumber: Int = FaceDetector.detect(imageBytes.getBytes, args(1))
      (text.toString, facesNumber)
    }}

    val faces: Array[(String, Int)] = faceCount.collect()

    faces.foreach(face => {
      println("Image name: " + face._1 + ", Face count: " + face._2)
    })
  }
}

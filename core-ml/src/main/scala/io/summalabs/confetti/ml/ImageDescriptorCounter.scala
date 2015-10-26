package io.summalabs.confetti.ml

import org.apache.hadoop.io.{BytesWritable, Text}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.opencv.core.Core

/**
 * @author Sergey Kovalev.
 */
object ImageDescriptorCounter {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ImageDecriptorCounter")
    val sc = new SparkContext(conf)

    val file = sc.sequenceFile(args(0), classOf[Text], classOf[BytesWritable])

    val imagesDescriptors: RDD[(String, Descriptor)] = file.map{image => {
      System.out.println("--------Library Path: " + System.getProperty("java.library.path"))
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME)

      val text = image._1.asInstanceOf[Text]
      val imageBytes = image._2.asInstanceOf[BytesWritable]
      val descriptor: Descriptor = Descriptor.buildDscFromRawData(imageBytes.getBytes, Descriptor.DEF_BIN_NUMBER, true)
      (text.toString, descriptor)
    }}

    val descriptors: Array[(String, Descriptor)] = imagesDescriptors.collect()

    descriptors.foreach(imageDesc => {
      println("Image name: " + imageDesc._1)
      println("Image descriptor size: " + imageDesc._2.getSize)
    })
  }

}

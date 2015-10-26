package io.summalabs.confetti.ml

import org.apache.hadoop.io.{BytesWritable, Text}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * @author Sergey Kovalev.
 */
object ImageDescriptorCounter {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ImageDecriptorCounter")
    val sc = new SparkContext(conf)

    val file = sc.sequenceFile(args(0), classOf[Text], classOf[BytesWritable])

    val imagesDescriptors: RDD[Descriptor] = file.map{image => {
      val text = image._1.asInstanceOf[Text]
      val imageBytes = image._2.asInstanceOf[BytesWritable]
      text.toString
      val descriptor: Descriptor = Descriptor.buildDscFromRawData(imageBytes.getBytes, Descriptor.DEF_BIN_NUMBER, true)
      descriptor
    }}

    val descriptors: Array[Descriptor] = imagesDescriptors.collect()



    println("Image descriptor size: " + descriptors(0).getSize)
  }
}

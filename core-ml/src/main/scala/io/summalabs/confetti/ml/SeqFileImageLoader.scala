package io.summalabs.confetti.ml

import org.apache.hadoop.io.{BytesWritable, Text}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.hipi.image.{HipiImage, HipiImageHeader}

/**
 * @author Sergey Kovalev.
 */
object SeqFileImageLoader {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ImageLoader")
    val sc = new SparkContext(conf)

    val file = sc.sequenceFile(args(0), classOf[Text], classOf[BytesWritable])

    val images: RDD[String] = file.map{image => {
      val text = image._1.asInstanceOf[Text]
      val imageBytes = image._2.asInstanceOf[BytesWritable]
      text.toString
    }}

    val text: Array[String] = images.take(1)

    println("Image text: " + text(0))
  }
}

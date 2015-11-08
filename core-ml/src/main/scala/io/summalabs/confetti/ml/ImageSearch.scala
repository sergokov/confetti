package io.summalabs.confetti.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * @author Sergey Kovalev.
 */
object ImageSearch {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ImageDescriptorSearch")
    val sc = new SparkContext(conf)

    val images:RDD[(String, Array[Byte])] = ImageDescriptor.loadImages(sc, args(0))

    val descriptors: RDD[(String, Descriptor)] = ImageDescriptor.calcImageDescriptor(images).cache()

    val imageDescPca: RDD[(String, Array[Double])] =
      ImageDescriptor.applyPcaToDescriptors(descriptors).map(d => (d._1, d._2.toArray))

    val randomImgToSearch:(String, Array[Double]) = imageDescPca.take(1)(0)

    val imageSetToSearch: RDD[(String, Array[Double])] =
      imageDescPca.filter(e => !e._1.equals(randomImgToSearch._1))

    val nearest5Img: Array[(String, Double)] = ImageDescriptor.findNearestNImages(imageSetToSearch, 5)

    nearest5Img.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })
  }
}

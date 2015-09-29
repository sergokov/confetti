package io.summalabs.confetti.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.conf.Configuration
import org.hipi.image.{FloatImage, HipiImageHeader}
import org.hipi.imagebundle.mapreduce.HibInputFormat


/**
 * @author Sergey Kovalev.
 */
object HipiImageLoader {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ImageLoader")
    val sc = new SparkContext(conf)

    val config = new Configuration()

    val images: RDD[(HipiImageHeader, FloatImage)] = sc.newAPIHadoopRDD(config,
      classOf[HibInputFormat],
      classOf[HipiImageHeader],
      classOf[FloatImage])
    
    print(images.count())
  }
}

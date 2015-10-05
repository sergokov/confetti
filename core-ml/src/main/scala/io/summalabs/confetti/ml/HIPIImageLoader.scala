package io.summalabs.confetti.ml

import org.apache.hadoop.fs.{Path, LocalFileSystem}
import org.apache.hadoop.hdfs.DistributedFileSystem
import org.apache.hadoop.mapreduce.Job
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.hadoop.conf.Configuration
import org.hipi.image.{HipiImage, FloatImage, HipiImageHeader}
import org.hipi.imagebundle.mapreduce.HibInputFormat

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat


/**
 * @author Sergey Kovalev.
 */
object HIPIImageLoader {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ImageLoader")
    val sc = new SparkContext(conf)

    val config = new Configuration()

    config.set(HibInputFormat.IMAGE_CLASS, classOf[FloatImage].getName)

    config.set("fs.hdfs.impl", classOf[DistributedFileSystem].getName)
    config.set("fs.file.impl", classOf[LocalFileSystem].getName)
    config.addResource(new Path("/usr/local/hadoop/etc/hadoop/core-site.xml"))
    config.set(FileInputFormat.INPUT_DIR, args(0))

    val images: RDD[(HipiImageHeader, HipiImage)] = sc.newAPIHadoopRDD(config,
      classOf[HibInputFormat],
      classOf[HipiImageHeader],
      classOf[HipiImage])

    print("Image Count: " + images.count())
  }
}

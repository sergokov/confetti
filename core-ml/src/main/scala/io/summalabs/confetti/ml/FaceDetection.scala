package io.summalabs.confetti.ml

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, LocalFileSystem}
import org.apache.hadoop.hdfs.DistributedFileSystem
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.bytedeco.javacpp.opencv_core._

import org.hipi.image.{HipiImage, HipiImageHeader, FloatImage}
import org.hipi.imagebundle.mapreduce.HibInputFormat
import org.hipi.opencv.OpenCVUtils

/**
 * @author Sergey Kovalev.
 */
object FaceDetection {
  def main(args: Array[String]): Unit = {
    System.out.println("--------Library Path: " + System.getProperty("java.library.path"))

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

    val faces: RDD[Int] = images.map(imgData => {
      val value = imgData._2.asInstanceOf[FloatImage]
      val imageMat: Mat = OpenCVUtils.convertRasterImageToMat(value)
      val facesCount: Int = FaceDetector.detect(imageMat, args(1))
      facesCount
    })

    val facesNumber: Int = faces.reduce(_ + _)

    println(f"Faces number: $facesNumber%d")
  }
}



package io.summalabs.confetti.ml

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{Path, LocalFileSystem}
import org.apache.hadoop.hdfs.DistributedFileSystem
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.bytedeco.javacpp.opencv_core
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
      val image = imgData._2.asInstanceOf[FloatImage]


      // Generate opencv data type based on input pixel array data type / number of bands
      val pixelArrayDataType: Int = image.getPixelArray.getDataType
      val numBands: Int = image.getNumBands
      val openCVType: Int =OpenCVUtils.generateOpenCVType(pixelArrayDataType, numBands)
      // Create output mat
      val mat: opencv_core.Mat = new opencv_core.Mat(image.getHeight, image.getWidth, openCVType)
      val matType: Int = mat.`type`
      val depth: Int = opencv_core.CV_MAT_DEPTH(matType)

      System.out.println("------------numBands: " + numBands)
      System.out.println("------------MatType: " + matType)
      System.out.println("------------Depth: " + depth)

      mat.convertTo(mat, opencv_core.CV_8U)

      val imageMat: Mat = OpenCVUtils.convertRasterImageToMat(image)
      imageMat.convertTo(imageMat, opencv_core.CV_8U)
      val depth2: Int = opencv_core.CV_MAT_DEPTH(matType)
      System.out.println("------------Depth2: " + depth2)

      val facesCount: Int = FaceDetector.detect(imageMat, args(1))
      facesCount
    })

    val facesNumber: Int = faces.reduce(_ + _)

    println(f"Faces number: $facesNumber%d")
  }
}



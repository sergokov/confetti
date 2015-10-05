package io.summalabs.confetti.ml

import org.apache.hadoop.fs.{Path, LocalFileSystem}
import org.apache.hadoop.hdfs.DistributedFileSystem
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

    print("Image Count: " + images.count() + " elements ")

    val imgAvg: RDD[FloatImage] = images.map(imgData => {
      val value = imgData._2.asInstanceOf[FloatImage]
      val w = value.getWidth
      val h = value.getHeight

      // Get pointer to image data
      val valData = value.getData
      // Initialize 3 element array to hold RGB pixel average
      val avgData = new Array[Float](3)

      for (j <- 0 to h) {
        for (i <- 0 to w) {
          avgData(0) += valData((j * w + i) * 3 + 0) // R
          avgData(1) += valData((j * w + i) * 3 + 1); // G
          avgData(2) += valData((j * w + i) * 3 + 2); // B
        }
      }

      // Create a FloatImage to store the average value
      val avg = new FloatImage(1, 1, 3, avgData)
      // Divide by number of pixels in image
      avg.scale(1.0f / (w * h))

      avg
    })

    val total = sc.accumulator(0)

    val image: FloatImage = imgAvg.reduce((a, b) => {
      a.add(b)
      total += 1
      a
    })

    if (total.value > 0) {
      // Normalize sum to obtain average
      image.scale(1.0f / total.value)
      // Assemble final output as string
      val avgData = image.getData

      val r = avgData(0)
      val g = avgData(1)
      val b = avgData(2)

      println(f"Average pixel value: $r%f $g%f $b%f")
    }
  }
}

package io.summalabs.confetti.ml

import java.util.Random

import org.apache.spark.mllib.linalg.{Vectors, Vector, Matrix}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
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

    val descs: RDD[(String, Vector)] = ImageDescriptor
      .calcImageDescriptor(images).map(desc => (desc._1, Vectors.dense(desc._2.getValue)))

    val descMatrix: RowMatrix = new RowMatrix(descs.map(des => des._2))
    val pca: Matrix = descMatrix.computePrincipalComponents(8)
    val descPcaMatrix: RowMatrix = descMatrix.multiply(pca)

    val imageDescPca: RDD[(String, Array[Double])] =
      ImageDescriptor.joinRdds(descs, descMatrix.rows).map(desc => (desc._1, desc._2.toArray))

    val imgToSearch:(String, Array[Double]) = imageDescPca.take(20)(new Random().nextInt(20))

    println("---------Sample image name----------: " + imgToSearch._1)

    val imageSetToSearch: RDD[(String, Array[Double])] =
      imageDescPca.filter(e => !e._1.equals(imgToSearch._1))

    val nearestImg: Array[(String, Double)] =
      ImageDescriptor.findNearestNImages(imageSetToSearch, imgToSearch._2, 20)

    nearestImg.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })
  }
}

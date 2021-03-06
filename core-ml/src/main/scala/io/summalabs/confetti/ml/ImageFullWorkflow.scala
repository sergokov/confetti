package io.summalabs.confetti.ml

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * @author Sergey Kovalev.
 */
object ImageFullWorkflow {
  
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("ImageDescriptorCalculation")
    val sc = new SparkContext(conf)

    val images:RDD[(String, Array[Byte])] = ImageDescriptor.loadImages(sc, args(0))

    val descs: RDD[(String, Vector)] = ImageDescriptor
      .calcImageDescriptor(images).map(desc => (desc._1, Vectors.dense(desc._2.getValue)))

    val descMatrix: RowMatrix = new RowMatrix(descs.map(des => des._2))
    val pca: Matrix = descMatrix.computePrincipalComponents(8)
    val descPcaMatrix: RowMatrix = descMatrix.multiply(pca)

    val imageDescPca: RDD[(String, Vector)] = ImageDescriptor.joinRdds(descs, descMatrix.rows)

    ImageDescriptor.saveImageDescriptors(args(1), imageDescPca)

    val loadedImgDescs: RDD[(String, Array[Double])] = ImageDescriptor.loadImageDescriptors(sc, args(1)).cache()

    val randomImgToSearch:(String, Array[Double]) = loadedImgDescs.take(1)(0)

    val imageSetToSearch: RDD[(String, Array[Double])] =
      loadedImgDescs.filter(e => !e._1.equals(randomImgToSearch._1))

    val nearest5Img: Array[(String, Double)] =
      ImageDescriptor.findNearestNImages(imageSetToSearch, randomImgToSearch._2, 5)

    nearest5Img.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })
  }
}

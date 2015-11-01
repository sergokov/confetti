package io.summalabs.confetti.ml

import org.apache.hadoop.io.{BytesWritable, Text}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.opencv.core.Core

/**
 * @author Sergey Kovalev.
 */
object ImageDescriptorCalculation {
  var conf:SparkConf = _
  var sc:SparkContext = _

  def main(args: Array[String]): Unit = {
    conf = new SparkConf().setAppName("ImageDescriptorCalculation")
    sc = new SparkContext(conf)

    val images:RDD[(String, Array[Byte])] = readImages(args(0))
    val descriptors: RDD[(String, Descriptor)] = images.map{image => calcImageDescriptor(image)}.cache()

    val descVector: RDD[Vector] = descriptors.map(descriptor => toVector(descriptor._2))

    val descMatrix: RowMatrix = new RowMatrix(descVector)
    val pcaMatrix: Matrix = descMatrix.computePrincipalComponents(8)
    val descPcaMatrix: RowMatrix = descMatrix.multiply(pcaMatrix)

    val imageIndexed: RDD[(Long, String)] = descriptors.zipWithIndex().map(zi => (zi._2, zi._1._1))

    val descPcaIndexed: RDD[(Long, Vector)] =
      descPcaMatrix.rows.zipWithIndex().map(zi => (zi._2, zi._1))

    val imageDscPca: RDD[(String, Vector)] = imageIndexed.join(descPcaIndexed).map(j => j._2)

    saveImageDescriptors(args(1), imageDscPca)

    val loadImgDescriptors: RDD[(String, Array[Double])] = loadImageDescriptors(args(1)).cache()

    val sampleImgDesc:(String, Array[Double]) = loadImgDescriptors.take(1)(0)

    val distance: RDD[(String, Double)] = loadImgDescriptors.filter(e => !e._1.equals(sampleImgDesc._1)).map(d => {
      val distanceL1: Double = new Descriptor(sampleImgDesc._2).distL1(new Descriptor(d._2))
      (d._1, distanceL1)
    })

    val nearest5: Array[(String, Double)] = distance.sortBy(_._2).take(5)

    nearest5.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })
  }

  def readImages(path:String):RDD[(String, Array[Byte])] = {
    sc.sequenceFile(path, classOf[Text], classOf[BytesWritable]).map(w => (w._1.toString, w._2.getBytes))
  }

  def calcImageDescriptor(image:(String, Array[Byte])): (String, Descriptor) = {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val descriptor: Descriptor = new Descriptor(image._2, Descriptor.DEF_BIN_NUMBER, true)
    (image._1, descriptor)
  }

  def toVector(descriptor: Descriptor): Vector = {
    Vectors.dense(descriptor.getValue.map(f => f.toDouble))
  }

  def saveImageDescriptors(path:String, descriptor:RDD[(String, Vector)]): Unit = {
    descriptor.map(d => (d._1, new DoubleArrayWritable(d._2.toArray))).saveAsSequenceFile(path)
  }

  def loadImageDescriptors(path:String): RDD[(String, Array[Double])] = {
    sc.sequenceFile(path, classOf[Text], classOf[DoubleArrayWritable]).map(d => (d._1.toString, d._2.toArray))
  }

}

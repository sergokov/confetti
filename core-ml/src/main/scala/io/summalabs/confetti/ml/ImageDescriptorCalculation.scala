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
    conf = new SparkConf().setAppName("ImageDecriptorCounter")
    sc = new SparkContext(conf)

    val images:RDD[(String, Array[Byte])] = readImages(args(0))
    val descriptor: RDD[(String, Descriptor)] = images.map{image => calcImageDescriptor(image)}

    val descVector: RDD[Vector] = descriptor.map(descriptor => toVector(descriptor._2))

    val descMatrix: RowMatrix = new RowMatrix(descVector)
    val pcaMatrix: Matrix = descMatrix.computePrincipalComponents(8)
    val descPcaMatrix: RowMatrix = descMatrix.multiply(pcaMatrix)

    val imageIndexed: RDD[(Long, String)] = descriptor.zipWithIndex().map(zi => (zi._2, zi._1._1))

    val descPcaIndexed: RDD[(Long, Vector)] =
      descPcaMatrix.rows.zipWithIndex().map(zi => (zi._2, zi._1))

    val imageDscPca: RDD[(String, Vector)] = imageIndexed.join(descPcaIndexed).map(j => j._2)

//    imageDscPca.saveAsObjectFile(args(1))
//    imageDscPca = sc.objectFile(args(1))

    saveImageDescriptors(args(1), imageDscPca)

    val desc: RDD[(String, Array[Double])] = loadImageDescriptors(args(1))

    desc.foreach(imageDesc => {
      println("Image name: " + imageDesc._1)
      println("Image descriptor size: " + imageDesc._2.toString)
    })
  }

  def readImages(path:String):RDD[(String, Array[Byte])] = {
    sc.sequenceFile(path, classOf[Text], classOf[BytesWritable]).map(w => (w._1.toString, w._2.getBytes))
  }

  def calcImageDescriptor(image:(String, Array[Byte])): (String, Descriptor) = {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val descriptor: Descriptor = Descriptor.buildDscFromRawData(image._2, Descriptor.DEF_BIN_NUMBER, true)
    (image._1, descriptor)
  }

  def toVector(descriptor: Descriptor): Vector = {
    Vectors.dense(descriptor.getData.map(f => f.toDouble))
  }

  def saveImageDescriptors(path:String, descriptor:RDD[(String, Vector)]): Unit = {
    descriptor.map(d => (d._1, new DoubleArrayWritable(d._2.toArray))).saveAsSequenceFile(path)
  }

  def loadImageDescriptors(path:String): RDD[(String, Array[Double])] = {
    sc.sequenceFile(path, classOf[Text], classOf[DoubleArrayWritable]).map(d => (d._1.toString, d._2.toArray))
  }

}

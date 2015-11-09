package io.summalabs.confetti.ml

import org.apache.hadoop.io.{BytesWritable, Text}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Matrix, Vectors, Vector}
import org.apache.spark.rdd.RDD
import org.opencv.core.Core

/**
 * @author Sergey Kovalev.
 */
object ImageDescriptor {
  
  def loadImages(sc:SparkContext, path:String):RDD[(String, Array[Byte])] = {
    sc.sequenceFile(path, classOf[Text], classOf[BytesWritable]).map(w => (w._1.toString, w._2.getBytes))
  }

  def calcImageDescriptor(images: RDD[(String, Array[Byte])]): RDD[(String, Descriptor)] = {
    images.map { image => {
      System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
      val descriptor: Descriptor = new Descriptor(image._2, Descriptor.DEF_BIN_NUMBER, true)
      (image._1, descriptor)
    }}
  }

  def applyPcaToDescriptors(descriptors: RDD[(String, Descriptor)]):RDD[(String, Vector)] = {
    val descVector: RDD[Vector] = descriptors.map(descriptor => Vectors.dense(descriptor._2.getValue))
    val descMatrix: RowMatrix = new RowMatrix(descVector)
    val pcaMatrix: Matrix = descMatrix.computePrincipalComponents(8)
    val descPcaMatrix: RowMatrix = descMatrix.multiply(pcaMatrix)

    val imageIndexed: RDD[(Long, String)] = descriptors.zipWithIndex().map(zi => (zi._2, zi._1._1))
    val descPcaIndexed: RDD[(Long, Vector)] = descPcaMatrix.rows.zipWithIndex().map(zi => (zi._2, zi._1))
    imageIndexed.join(descPcaIndexed).map(j => j._2)
  }

  def saveImageDescriptors(path:String, descriptor:RDD[(String, Vector)]): Unit = {
    descriptor.map(d => (d._1, new DoubleArrayWritable(d._2.toArray))).saveAsSequenceFile(path)
  }

  def loadImageDescriptors(sc:SparkContext, path:String): RDD[(String, Array[Double])] = {
    sc.sequenceFile(path, classOf[Text], classOf[DoubleArrayWritable]).map(d => (d._1.toString, d._2.toArray))
  }

  def findNearestNImages(imgDescs: RDD[(String, Array[Double])], searchDesc:Array[Double], n:Int):Array[(String, Double)] = {
    val distance: RDD[(String, Double)] = imgDescs.map(d => {
      val distanceL1: Double = new Descriptor(searchDesc).distL1(new Descriptor(d._2))
      (d._1, distanceL1)
    })

    distance.sortBy(_._2).take(n)
  }
}

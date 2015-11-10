import java.io.File

import io.summalabs.confetti.ml.{DescriptorTest, Descriptor, ImageDescriptor}
import org.apache.commons.io.IOUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatest.junit.JUnitRunner

import scala.collection.mutable.ListBuffer

/**
 * @author sergo
 */

@RunWith(classOf[JUnitRunner])
class DescriptorCalculationTest extends FunSuite with Matchers with BeforeAndAfterAll  {
  private val path = "/Yandex.Shad_ImVideo_Competition1_anon/comp1_train/"
  private val folder: String = classOf[DescriptorTest].getResource(path).getFile
  private val files: Array[File] = new File(folder).listFiles.filter(_.getName.endsWith(".png"))
  private val imagesList = new ListBuffer[(String, Array[Byte])]()

  files.foreach(file => {
    val array: Array[Byte] =
      IOUtils.toByteArray(classOf[DescriptorTest].getResourceAsStream(path + file.getName))
    imagesList += ((file.getName, array))
  })

  val sparkConf = new SparkConf()
    .setAppName("DescriptorCalculationTest")
    .setMaster("local[2]")

  val sparkContext = new SparkContext(sparkConf)

  override def afterAll() {
    sparkContext.stop()
  }

  test("find nearest 5 image") {
    val images:RDD[(String, Array[Byte])] = sparkContext.parallelize[(String, Array[Byte])](imagesList)

    val descriptors: RDD[(String, Descriptor)] = ImageDescriptor.calcImageDescriptor(images).cache()

    val imageDescPca: RDD[(String, Array[Double])] =
      ImageDescriptor.applyPcaToDescriptors(descriptors).map(d => (d._1, d._2.toArray))

    val imgToSearch:(String, Array[Double]) = imageDescPca.take(1)(0)

    val imageSetToSearch: RDD[(String, Array[Double])] = imageDescPca.filter(e => !e._1.equals(imgToSearch._1))

    val nearest5Img: Array[(String, Double)] = ImageDescriptor.findNearestNImages(imageSetToSearch, imgToSearch._2, 5)

    nearest5Img.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })

    nearest5Img should not contain imgToSearch
  }

  test("find nearest 5 image without PCA") {
    val images:RDD[(String, Array[Byte])] = sparkContext.parallelize[(String, Array[Byte])](imagesList)

    val descriptors: RDD[(String, Descriptor)] = ImageDescriptor.calcImageDescriptor(images).cache()

    val imageDescPca: RDD[(String, Array[Double])] = descriptors.map(d => (d._1, d._2.getValue.toArray))

    val imgToSearch:(String, Array[Double]) = imageDescPca.take(1)(0)

    val imageSetToSearch: RDD[(String, Array[Double])] = imageDescPca.filter(e => !e._1.equals(imgToSearch._1))

    val nearest5Img: Array[(String, Double)] = ImageDescriptor.findNearestNImages(imageSetToSearch, imgToSearch._2, 5)

    nearest5Img.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })

    nearest5Img should not contain imgToSearch
  }
}

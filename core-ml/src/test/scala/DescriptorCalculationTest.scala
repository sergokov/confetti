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

  test("test reduce by") {
    val testSet:Array[(String, Double)] =
      Array(("img1", 0.723d), ("img2", 0.625d), ("img3", 0.524d), ("img4", 0.834d),
        ("img5", 0.321d), ("img6", 0.148d), ("img7", 1.524d), ("img8", 0.924d), ("img9", 0.21d), ("img10", 3.524d))

    val testSetRdd:RDD[(String, Double)] = sparkContext.parallelize[(String, Double)](testSet)

    val nearest5: Array[(String, Double)] = testSetRdd.sortBy(_._2).take(5)

    nearest5.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })
  }

  test("test reduce by 2") {
    val testSet1:Array[(String, Double)] =
      Array(("img10", 0.723d), ("img20", 0.625d), ("img30", 0.524d), ("img40", 0.834d),
        ("img50", 0.321d), ("img60", 0.148d), ("img70", 1.524d), ("img80", 0.924d), ("img90", 0.21d), ("img100", 3.524d))

    val testSet2:Array[(String, Double)] =
      Array(("img11", 0.723d), ("img21", 0.625d), ("img31", 0.524d), ("img41", 0.834d),
        ("img51", 0.321d), ("img61", 0.148d), ("img71", 1.524d), ("img81", 0.924d), ("img91", 0.21d), ("img101", 3.524d))

    val testSetRdd1:RDD[(String, Double)] = sparkContext.parallelize[(String, Double)](testSet1, 3)

    val testSetRdd2:RDD[(String, Double)] = sparkContext.parallelize[(String, Double)](testSet2, 3)

    val index1: RDD[(Long, (String, Double))] = testSetRdd1.zipWithIndex().map(m => {(m._2, m._1)})

    val index2: RDD[(Long, (String, Double))] = testSetRdd2.zipWithIndex().map(m => {(m._2, m._1)})

    val join: RDD[(Long, ((String, Double), (String, Double)))] = index1.join(index2)

    val collect: Array[(Long, ((String, Double), (String, Double)))] = join.collect()

    val nearest5: Array[(String, Double)] = testSetRdd1.sortBy(_._2).take(5)

    nearest5.foreach(el => {
      println("Image name: " + el._1)
      println("Image descriptor distance L1: " + el._2)
    })
  }

}

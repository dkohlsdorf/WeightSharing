package experiments

import autodiff.Tensor
import breeze.linalg.{DenseMatrix, reshape}
import breeze.plot._
import nn.FeedForwardSigmoid

import scala.io.Source
import scala.util.Random


object MNIST {

  /**
    * load mnist training set
    */
  final val mnistTrain = Random.shuffle(Source.fromFile("data/mnist_train.csv").getLines().map (line => {
    val cmp = line.trim().split(",")
    (DenseMatrix(cmp.slice(1, cmp.length).map(_.toDouble)), cmp(0).toDouble.toInt)
  }).toList)

  /**
    * load mnist testing set
    */
  final val mnistTest = Random.shuffle(Source.fromFile("data/mnist_test.csv").getLines().map (line => {
    val cmp = line.trim().split(",")
    (DenseMatrix(cmp.slice(1, cmp.length).map(_.toDouble)), cmp(0).toDouble.toInt)
  }).toList)

  /**
    * scale weights back to 28 x 28 image
    */
  def toImg(weights: DenseMatrix[Double], i: Int): DenseMatrix[Double] = reshape(weights(::, i), 28, 28)

  /**
    * plot weights
    */
  def plot(weights: DenseMatrix[Double]): Unit = {
    for (i <- 0 until 100) {
      val f1 = Figure()
      f1.subplot(0) += image(toImg(weights, i))
      f1.saveas(s"image_multi${i}.png")
    }
  }

  def main(args: Array[String]): Unit = {
    // define tensors
    val x   = new Tensor("in")
    val w1  = new Tensor("w1")
    val b1  = new Tensor("b1")
    val w2  = new Tensor("w2")
    val b2  = new Tensor("b2")

    // setup weights
    x   := DenseMatrix.rand[Double](1,   784)
    w1  := (DenseMatrix.rand[Double](784, 100) - 0.5) * 0.001
    b1  := DenseMatrix.zeros[Double](1,   100)
    w2  := (DenseMatrix.rand[Double](100, 10)  - 0.5) * 0.001
    b2  := DenseMatrix.zeros[Double](1,   10)

    // build to layer nn
    val layer1 = new FeedForwardSigmoid(x, w1, b1)
    val layer2 = new FeedForwardSigmoid(layer1.Layer, w2, b2)

    var total = 0.0
    for {
      j      <- 0 until 10 // epochs
      ((i, y), k) <- mnistTrain.zipWithIndex
    } {
      x := i / i.max

      val correct    = DenseMatrix.zeros[Double](1, 10)
      correct(0, y)  = 1.0

      // forward pass
      val prediction = layer2.Layer.fwd
      val error      = -(correct - prediction)

      // learn
      layer2.Layer.bwd(layer2.Layer.name, error)
      layer2.update(0.01)
      layer1.update(0.01)
      layer2.Layer.reset

      total += 0.5 * error.map(x => math.pow(x, 2)).sum
      if(k == mnistTrain.size - 1) {
        println(total / mnistTrain.size.toDouble)
        total = 0.0
      }
    }

    plot(w1.fwd)

    // evaluations
    var c = 0.0
    for ((i, y) <- mnistTest) {
      x := i / i.max
      layer2.Layer.reset
      c = c + (if ( (layer2.Layer.fwd.t.argmax._1) == y ) 1.0 else 0.0)
    }
    println("Accuracy: " + c / mnistTest.size.toDouble)
  }
}

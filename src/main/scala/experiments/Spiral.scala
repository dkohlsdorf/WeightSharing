package experiments

import autodiff.Tensor
import breeze.linalg.DenseMatrix
import nn.FeedForwardSigmoid

import scala.io.Source
import scala.util.Random

/**
  * 2 layer 3 spirals experiment
  *
  * by Daniel Kohlsdorf
  */
object Spiral {

  /**
    * Create the three spiral data
    */
  final val spiral = Random.shuffle(Source.fromFile("data/spiral.csv").getLines().map (line => {
    val cmp = line.trim().split(",")
    (DenseMatrix(cmp.slice(0, 2).map(_.toDouble)), cmp(2).toDouble.toInt)
  }).toList)

  def main(args: Array[String]): Unit = {
    // define tensors
    val input = new Tensor("x")
    val W1    = new Tensor("w1")
    val B1    = new Tensor("w2")
    val W2    = new Tensor("w3")
    val B2    = new Tensor("w4")

    // split data into 80 train and 20 test
    val train = spiral.slice(0, (spiral.size * 0.8).toInt)
    val test  = spiral.slice((spiral.size * 0.2).toInt, spiral.size)

    // initialize the weights and shapes
    input :=  DenseMatrix.rand[Double](1,   2)
    W1    := (DenseMatrix.rand[Double](2, 100) - 0.5) * 0.01
    B1    :=         DenseMatrix.zeros(1, 100)
    W2    := (DenseMatrix.rand[Double](100, 3) - 0.5) * 0.01
    B2    :=         DenseMatrix.zeros(1,   3)

    // 2 layer network
    val layer1 = new FeedForwardSigmoid(input, W1, B1)
    val layer2 = new FeedForwardSigmoid(layer1.Layer, W2, B2)

    for {
      i <- 0 until 2500
    } {
      var total = 0.0
      for ((x, y) <- train) {
        val correct    = DenseMatrix.zeros[Double](1, 3)
        correct(0, y)  = 1.0

        // forward pass through complete network
        input := x
        val prediction = layer2.Layer.fwd

        // compute error and compute all gradients
        // also update weights then reset all values
        val error = -(correct - prediction)
        layer2.Layer.bwd(layer2.Layer.name, error)
        layer2.update(1.0)
        layer1.update(1.0)
        layer2.Layer.reset

        total += 0.5 * (correct - prediction).map(x => math.pow(x, 2)).sum
      }
      if(i % 1000 == 0) println(total / 1000)
    }

    // compute accuracy
    var c = 0.0
    for ((x, y) <- test) {
      input := x
      layer2.Layer.reset
      c = c + (if ( (layer2.Layer.fwd.t.argmax._1) == y ) 1.0 else 0.0)
    }
    println("Acc: " + c / test.size.toDouble)
  }

}

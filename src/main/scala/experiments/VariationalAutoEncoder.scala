package experiments

import autodiff._
import breeze.linalg.{DenseMatrix, DenseVector, reshape}
import breeze.plot._
import breeze.numerics._
import nn.{FeedForwardSigmoid, FeedForwardTanH}

import scala.io.Source
import scala.util.Random

/**
  * A variational auto encoder
  *
  * 1) Predict hidden layer
  * 2) Predict Mean and Variance
  * 3) s ~ Sample Uniform N(0, I)
  * 4) x = var * s + mean
  * 5) Predict output from x
  * 6) Measure error to input and backprop
  *
  * [1] https://arxiv.org/pdf/1606.05908.pdf
  *
  * By Daniel Kohlsdorf
  **/

object VariationalAutoEncoder {
  /**
    * The VAE distribution
    */
  val gauss = 25
  val gaussian = breeze.stats.distributions.MultivariateGaussian(DenseVector.zeros[Double](gauss), DenseMatrix.eye[Double](gauss))

  /**
    * Initialization distribution
    */
  val std = 0.01
  val samplePDF = breeze.stats.distributions.Gaussian(0, math.sqrt(std))

  /**
    * Sample truncated normal
    */
  def truncatedNormal(std: Double): Double = {
    var x = samplePDF.draw()
    while (math.abs(x) > 2.0 * std) {
      x = samplePDF.draw()
    }
    x
  }

  /**
    * Sample truncated normal matrix
    */
  def rand(n: Int, m: Int) = DenseMatrix.zeros[Double](n,m).map(x => truncatedNormal( math.sqrt(std) ))

  /**
    * Read some MNIST
    */
  final val mnistTrain = Random.shuffle(Source.fromFile("data/mnist_train.csv").getLines().map (line => {
    val cmp = line.trim().split(",")
    (DenseMatrix(cmp.slice(1, cmp.length).map(_.toDouble)), gaussian.draw().toDenseMatrix)
  }).toList).take(200)

  /**
    * scale weights back to 28 x 28 image
    */
  def toImg(weights: DenseMatrix[Double]): DenseMatrix[Double] = reshape(weights, 28, 28)

  /**
    * plot weights
    */
  def plot(weight: DenseMatrix[Double], i: Int): Unit = {
    val f1 = Figure()
    f1.subplot(0) += image(toImg(weight))
    f1.saveas(s"image_multi${i}.png")
  }

  def main(args: Array[String]): Unit = {
    val h = 50

    // setup tensors
    val in     = new Tensor("in")
    val wenc   = new Tensor("wenc")
    val benc   = new Tensor("benc")
    val wmean  = new  Tensor("wmean")
    val bmean  = new Tensor("bmean")
    val wsig  = new Tensor("wsig")
    val bsig  = new Tensor("bsig")
    val wdec   = new Tensor("wdec")
    val bdec   = new Tensor("bdec")
    val sample = new Tensor("sample")
    val half = new Tensor("0_5")

    // setup variational auto encoder
    val enc   = new FeedForwardTanH(in, wenc, benc)
    val mean1 = new Inner(enc.Layer, wmean)
    val mean  = new Add(mean1, bmean)
    val sigm1  = new Inner(enc.Layer, wsig)
    val sigm2  = new Add(sigm1, bsig)
    val sigm   = new Mul(sigm2, half)
    val d1 = new Exp(sigm)
    val d2 = new Mul(sample, d1)
    val dist = new Add(mean, d2)
    val dec  = new FeedForwardSigmoid(dist, wdec, bdec)

    // register all variables
    enc.Layer.register(mean1.name)
    wmean.register(mean1.name)
    mean1.register(mean.name)
    bmean.register(mean.name)
    enc.Layer.register(sigm1.name)
    wsig.register(sigm1.name)
    sigm1.register(sigm2.name)
    bsig.register(sigm2.name)
    sigm2.register(sigm.name)
    half.register(sigm.name)
    sigm.register(d1.name)
    sample.register(d2.name)
    d1.register(d2.name)
    mean.register(dist.name)
    d2.register(dist.name)

    // initialize all weights
    wenc  := rand(784, h)
    benc  := DenseMatrix.zeros[Double](1,   h)
    wmean := rand(h,  gauss)
    bmean := DenseMatrix.zeros[Double](1,  gauss)
    wsig  := rand(h,  gauss)
    bsig  := DenseMatrix.zeros[Double](1,  gauss)
    wdec  := rand(gauss, 784)
    bdec  := DenseMatrix.zeros[Double](1, 784)
    half  := DenseMatrix.ones[Double](1,  gauss) * 0.5

    // plot graph for debugging
    dec.Layer.graph("").distinct.foreach(println)

    for(epoch <- 0 until 250) {
      var total = 0.0
      for ((x, s) <- mnistTrain) {
        // forward pass
        in := x / x.max
        sample := s
        val y = dec.Layer.fwd

        // auto encoder error
        val error = -(in.fwd - y)

        // backwards pass
        dec.Layer.bwd("err", error)
        mean.bwd("err", mean.fwd)
        sigm.bwd("err", exp(2.0 * sigm.fwd) + 1.0)

        // compute loss
        val KLD = 0.5 * ((sigm.fwd - ( pow(mean.fwd , 2.0) - exp(2.0 * sigm.fwd) ) + 1.0)).sum
        total +=  0.5 * error.map(x => math.pow(x, 2)).sum + KLD

        // update mean, sigma, decoder and encoder
        val rate = 0.001
        dec.update(rate)
        enc.update(rate)
        wmean := wmean.fwd - (rate * wmean.fwd)
        wsig  := wsig.fwd  - (rate * wsig.fwd)
        bmean := bmean.fwd - (rate * bmean.fwd)
        bsig  := bsig.fwd  - (rate * bsig.fwd)

        // reset network
        mean.reset
        sigm.reset
        enc.Layer.reset
        dec.Layer.reset
      }
      println(total / mnistTrain.size.toDouble)
    }

    // generate from samples
    val testdec  = new FeedForwardSigmoid(sample, wdec, bdec)
    for {i <- 0 until 20} {
      sample := gaussian.draw().toDenseMatrix
      plot(testdec.Layer.fwd, i)
      testdec.Layer.reset
    }

  }
}

package experiments

import autodiff.{Tensor, Node}
import breeze.linalg.DenseMatrix
import nn.{FeedForwardTanH, LSTM}

object Trigonometric {

  /**
    * Initialization Distribution
    */
  val std = 1.0
  val gaussian = breeze.stats.distributions.Gaussian(0, math.sqrt(std))

  /**
    * Sample a truncated normal
    */
  def truncatedNormal(std: Double): Double = {
    var x = gaussian.draw()
    while (math.abs(x) > 2.0 * std) {
      x = gaussian.draw()
    }
    x
  }

  /**
    * Sample initial matrix
    */
  def rand(n: Int, m: Int) = DenseMatrix.zeros[Double](n, m).map(x => truncatedNormal(math.sqrt(std)))

  /**
    * Setup weights
    */
  val wf = new Tensor("wforget")
  val bf = new Tensor("bforget")
  val wi = new Tensor("win")
  val bi = new Tensor("bin")
  val wo = new Tensor("woff")
  val bo = new Tensor("boff")
  val wc = new Tensor("wcell")
  val bc = new Tensor("bcell")
  val wl = new Tensor("w")
  val bl = new Tensor("b")
  val h = new Tensor("hidden")
  val c = new Tensor("cell")

  /**
    * The cos() , sin() dataset
    */
  val omega = 2 * math.Pi * 0.05

  def trig(n: Int): List[DenseMatrix[Double]] = (for (i <- 0 to n) yield DenseMatrix(math.cos(i * omega), math.sin(i * omega)).t).toList

  /**
    * Unroll the lastm for as many steps as input
    * also each step has an output layer
    */
  def unroll(in: List[Node]): (LSTM, List[FeedForwardTanH]) =
    if (in.size == 1) {
      val lstm = new LSTM(in.head, h, c, wf, bf, wi, bi, wo, bo, wc, bc)
      val classify = new FeedForwardTanH(lstm.Layer._2, wl, bl)
      (lstm, List(classify))
    } else {
      val (unrolled, out) = unroll(in.tail)
      val lstm = new LSTM(in.head, unrolled.Layer._2, unrolled.Layer._1, wf, bf, wi, bi, wo, bo, wc, bc)
      val classify = new FeedForwardTanH(lstm.Layer._2, wl, bl)
      (lstm, classify :: out)
    }


  def main(args: Array[String]): Unit = {
    val hidden = 15

    // initialize weights and create data
    wf := rand(hidden + 2, hidden)
    bf := (DenseMatrix.zeros[Double](1, hidden))
    wi := rand(hidden + 2, hidden)
    bi := (DenseMatrix.zeros[Double](1, hidden))
    wo := rand(hidden + 2, hidden)
    bo := (DenseMatrix.zeros[Double](1, hidden))
    wc := rand(hidden + 2, hidden)
    bc := (DenseMatrix.zeros[Double](1, hidden))
    wl := rand(hidden, 2)
    bl := (DenseMatrix.zeros[Double](1, 2))
    h := DenseMatrix.zeros[Double](1, hidden)
    c := DenseMatrix.zeros[Double](1, hidden)

    val seq = trig(500)

    // unroll for 5 steps
    val steps = 5
    val in = (for {i <- 0 until steps} yield {
      val x = new Tensor(s"input${steps - i - 1}")
      x
    }).toList
    val (lstm, classify) = unroll(in)

    // plot graph for debugging
    lstm.Layer._2.graph("").distinct.foreach(println)

    var t = 0
    val epochs = 150
    for {i <- 0 until epochs} {
      var total = 0.0

      // reset initial hidden activation and cell state
      h := DenseMatrix.zeros[Double](1, hidden)
      c := DenseMatrix.zeros[Double](1, hidden)
      for {
        j <- 0 until 100 - steps + 1
      } {
        t += 1

        //  build input and output
        val w = seq.slice(j, j + steps).reverse
        val l = seq.slice(j + 1, j + steps + 1).reverse

        // set inputs
        for (k <- 0 until steps) in(k) := w(k)

        // set output errors
        total += (for (k <- 0 until steps) yield {
          val y = classify(k).Layer.fwd
          val error = -(l(k) - y)
          classify(k).Layer.bwd("error", error)
          (0.5 * error.map(x => math.pow(x, 2)).sum)
        }).sum

        // learning
        val rate = 0.01
        lstm.update(rate)
        classify(0).update(rate)

        // reset all
        lstm.Layer._2.reset
        lstm.Layer._1.reset
        classify.foreach(_.Layer.reset)
        h := lstm.Layer._2.fwd
        c := lstm.Layer._1.fwd
      }
      println(total)
    }

    val w = seq.slice(0, steps)
    for (k <- 0 until steps) in(k) := w(k)

    for {
      j <- 0 until 100
    } {
      val y = classify.head.Layer.fwd
      lstm.Layer._2.reset
      lstm.Layer._1.reset
      classify.foreach(_.Layer.reset)
      println(s"${y(0, 0)} ${y(0, 1)}")
      for (k <- 1 until steps) {
        in(k) := in(k - 1).fwd
      }
      in(0) := y
      h := lstm.Layer._2.fwd
      c := lstm.Layer._1.fwd
    }

  }


}

package autodiff

import breeze.linalg.DenseMatrix
import breeze.numerics._


class Tanh(x: Node) extends Node(Node.getName, Vector(x)) {

  def typename = "tanh"

  def compute(x: Vector[DenseMatrix[Double]]): DenseMatrix[Double] = {
    (exp(x(0)) - exp(-x(0))) :/ (exp(x(0)) + exp(-x(0)))
  }

  def distribute(gradient: DenseMatrix[Double]): Unit = {
    val ones = DenseMatrix.ones[Double](fwd.rows, fwd.cols)
    val derivative = (ones - pow(fwd, 2))
    x.bwd(name, gradient :* derivative)
  }

}

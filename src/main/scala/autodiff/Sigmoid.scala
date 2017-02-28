package autodiff

import breeze.linalg.DenseMatrix
import breeze.numerics.sigmoid

class Sigmoid(x: Node) extends Node(Node.getName, Vector(x)) {

  def typename = "sigmoid"

  def compute(x: Vector[DenseMatrix[Double]]): DenseMatrix[Double] = sigmoid(x(0))

  def distribute(gradient: DenseMatrix[Double]): Unit = {
    val ones = DenseMatrix.ones[Double](fwd.rows, fwd.cols)
    val derivative = (ones - fwd) :* fwd
    x.bwd(name, gradient :* derivative)
  }

}

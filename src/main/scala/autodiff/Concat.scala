package autodiff

import breeze.linalg.{DenseMatrix}

class Concat(x: Node, y: Node) extends Node(Node.getName, Vector(x, y)) {

  def typename = "concat"

  def compute(x: Vector[DenseMatrix[Double]]): DenseMatrix[Double] = {
    DenseMatrix.horzcat(x(0), x(1))
  }

  def distribute(gradient: DenseMatrix[Double]): Unit = {
    x.bwd(name, gradient(::,          0 until x.fwd.cols))
    y.bwd(name, gradient(::, x.fwd.cols until gradient.size))
  }

}

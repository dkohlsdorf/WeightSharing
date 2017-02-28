package autodiff

import breeze.linalg.DenseMatrix

class Inner(x: Node, y: Node) extends Node(Node.getName, Vector(x, y)) {

  def typename = "inner"

  def compute(x: Vector[DenseMatrix[Double]]): DenseMatrix[Double] = x(0) * x(1)

  def distribute(gradient: DenseMatrix[Double]): Unit = {
    x.bwd(name, gradient * y.fwd.t)
    y.bwd(name, x.fwd.t * gradient)
  }

}

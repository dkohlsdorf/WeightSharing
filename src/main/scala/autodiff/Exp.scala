package autodiff

import breeze.linalg.DenseMatrix

class Exp(x: Node) extends Node(Node.getName, Vector(x)) {

  def typename = "exp"

  def compute(x: Vector[DenseMatrix[Double]]): DenseMatrix[Double] = {
    x(0).map(a => math.exp(a))
  }

  def distribute(gradient: DenseMatrix[Double]): Unit = {
    x.bwd(name, fwd)
  }

}



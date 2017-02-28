package autodiff

import breeze.linalg.DenseMatrix

class Tensor(name: String) extends Node(name, Vector.empty[Node]) {

  def typename = "tensor"

  def :=(x: DenseMatrix[Double]): Unit = {
    Output = Some(x)
  }

  def compute(x: Vector[DenseMatrix[Double]]): DenseMatrix[Double] ={
    if(Output == None) println(s"ERROR: ${name}")
    Output.get
  }

  def distribute(gradient: DenseMatrix[Double]): Unit = { }

  override def reset = {
    Gradients.clear()
  }

}

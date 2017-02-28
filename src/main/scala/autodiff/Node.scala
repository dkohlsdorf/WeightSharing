package autodiff

import breeze.linalg.{DenseMatrix}

import scala.collection.mutable

abstract class Node(val name: String, inputs: Vector[Node]) {
  /**
    * Names of parents who have to send a gradient before backwards
    * pass continues
    */
  val WaitFor   = mutable.Set.empty[String]

  /**
    * All parents gradients
    */
  val Gradients = mutable.HashMap.empty[String, DenseMatrix[Double]]

  /**
    * Set value of forwards computation
    */
  protected var Output: Option[DenseMatrix[Double]] = None

  /**
    * Should attach node type to name
    */
  def typename: String

  /**
    * Recursively plot parent child relationships in dot format.
    * Also output error if the waitlist does not include the expected parent
    */
  def graph(parent: String): List[String] = {
    if(!WaitFor.contains(parent) && parent.size > 1) throw new Exception(s"${parent} ${typename}_${name} connection not found")
    if(inputs.size == 0) List.empty[String]
    else {
      (for (in <- inputs) yield (s"${in.typename}_${in.name} -> ${typename}_${name};" :: in.graph(name))).flatten.toList
    }
  }

  def compute(x: Vector[DenseMatrix[Double]]): DenseMatrix[Double]

  def distribute(gradient: DenseMatrix[Double]): Unit

  def gradient: DenseMatrix[Double] = Gradients.map(_._2).reduce(_ + _)

  def register(parent: String): Unit = WaitFor += parent

  def reset: Unit = {
    Gradients.clear()
    Output = None
    inputs.foreach(_.reset)
  }

  def fwd: DenseMatrix[Double] = Output match {
    case Some(x) => x
    case None => {
      val x = compute(inputs.map(_.fwd))
      Output = Some(x)
      x
    }
  }

  def ready = (WaitFor & Gradients.keySet) == WaitFor

  def bwd(parent: String, gradient: DenseMatrix[Double]): Unit = {
    Gradients += parent -> gradient
    if(ready) distribute(gradient)
  }

}

object Node {

  final var CurrentID = -1

  def getName: String = {
    CurrentID += 1
    s"node_${CurrentID}"
  }

}

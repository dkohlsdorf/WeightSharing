package nn

import autodiff._

class FeedForwardTanH(x: Node, w: Tensor, b: Tensor) {

  private def mul: Node = {
    val dot = new Inner(x, w)
    x.register(dot.name)
    w.register(dot.name)
    dot
  }

  private def add: Node = {
    val dot = mul
    val sum = new Add(dot, b)
    dot.register(sum.name)
    b.register(sum.name)
    sum
  }

  private def tanh = {
    val sum = add
    val activation = new Tanh(sum)
    sum.register(activation.name)
    activation
  }

  final val Layer = tanh

  def update(rate: Double): Unit = {
    w := w.fwd - (rate * (w.gradient))
    b := b.fwd - (rate * (b.gradient))
  }

}

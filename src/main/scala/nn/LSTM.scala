package nn

import autodiff._
import breeze.linalg.DenseMatrix

import scala.collection.mutable

class LSTM(val in: Node, val ht: Node, val ct: Node, wf: Tensor, bf: Tensor, wi: Tensor, bi: Tensor, wo: Tensor, bo: Tensor, wc: Tensor, bc: Tensor) {

  val x = new Concat(in ,ht)
  val input  = new FeedForwardSigmoid(x, wi, bi)
  val output = new FeedForwardSigmoid(x, wo, bo)
  val forget = new FeedForwardSigmoid(x, wf, bf)
  val cell   = new FeedForwardTanH(x, wc, bc)

  val forgetCell = new Mul(forget.Layer, ct)
  val inCell     = new Mul(input.Layer, cell.Layer)
  val cellState  = new Add(inCell, forgetCell)
  val cellAct    = new Tanh(cellState)
  val hidden     = new Mul(output.Layer, cellAct)

  final val Layer = {
    in.register(x.name)
    ht.register(x.name)

    forget.Layer.register(forgetCell.name)
    ct.register(forgetCell.name)

    input.Layer.register(inCell.name)
    cell.Layer.register(inCell.name)

    inCell.register(cellState.name)
    forgetCell.register(cellState.name)

    cellState.register(cellAct.name)

    output.Layer.register(hidden.name)
    cellAct.register(hidden.name)

    (cellState, hidden)
  }

  def update(rate: Double): Unit = {
      wf := wf.fwd - ((rate * wf.gradient))
      bf := bf.fwd - ((rate * bf.gradient))
      wi := wi.fwd - ((rate * wi.gradient))
      bi := bi.fwd - ((rate * bi.gradient))
      wc := wc.fwd - ((rate * wc.gradient))
      bc := bc.fwd - ((rate * bc.gradient))
      wo := wo.fwd - ((rate * wo.gradient))
      bo := bo.fwd - ((rate * bo.gradient))
  }

}

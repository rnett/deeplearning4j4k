package com.rnett.deeplearning4j4k.builders

import com.rnett.deeplearning4j4k.NNConfDSL
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer
import java.util.*


class LayersBuilder(globalConfig: NeuralNetConfiguration.Builder,
                    layerMap: Map<Int, NeuralNetConfiguration.Builder>) :
    NeuralNetConfiguration.ListBuilder(globalConfig, layerMap) {
    constructor(globalConfig: NeuralNetConfiguration.Builder) : this(globalConfig, HashMap<Int, NeuralNetConfiguration.Builder>())

    @NNConfDSL
    operator fun <L : Layer> L.unaryPlus(): L{
        layer(this)
        return this
    }

    @NNConfDSL
    operator fun <L : Layer.Builder<L>> L.unaryPlus(): Layer {
        val l = this.build<Layer>()
        layer(l)
        return l
    }

    @NNConfDSL
    inline infix fun <L : Layer> L.addWith(builder: L.() -> Unit) = +apply(builder)

    @NNConfDSL
    inline infix fun <L : Layer.Builder<L>> L.addWith(builder: L.() -> Unit) = +apply(builder)

    @NNConfDSL
    infix fun <L : Layer> Int.layerAt(layer: L): L{
        layer(this, layer)
        return layer
    }

    @NNConfDSL
    operator fun set(index: Int, layer: Layer) = index layerAt layer

    @NNConfDSL
    inline infix fun BaseRecurrentLayer.makeBidirectional(mode: Bidirectional.Mode) =
        +Bidirectional.Builder(mode, this).build()

    @NNConfDSL
    inline infix fun LastTimeStep.makeBidirectional(mode: Bidirectional.Mode) =
        +Bidirectional.Builder(mode, this).build()

    @NNConfDSL
    inline infix fun BaseWrapperLayer.makeBidirectional(mode: Bidirectional.Mode) =
        +Bidirectional.Builder(mode, this).build()

    @NNConfDSL
    inline infix fun Layer.makeFrozen(mode: Bidirectional.Mode) =
        +FrozenLayer.Builder().apply { layer(this@makeFrozen) }.build()


}
package com.rnett.deeplearning4j4k.builders

import com.rnett.deeplearning4j4k.NNConfDSL
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.Layer
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
    infix fun <L : Layer> L.addWith(builder: L.() -> Unit) = +apply(builder)

    @NNConfDSL
    infix fun <L : Layer> Int.layerAt(layer: L): L{
        layer(this, layer)
        return layer
    }

    @NNConfDSL
    operator fun set(index: Int, layer: Layer) = index layerAt layer
}
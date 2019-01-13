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


class LayersBuilder(
    val globalConfig: NeuralNetConfiguration.Builder,
    layerMap: Map<Int, NeuralNetConfiguration.Builder>) :
    NeuralNetConfiguration.ListBuilder(globalConfig, layerMap) {
    constructor(globalConfig: NeuralNetConfiguration.Builder) : this(globalConfig, HashMap<Int, NeuralNetConfiguration.Builder>())


    //private var lastNOut: Long = 0

    private fun addLayer(layer: Layer, index: Int? = null) {
        /*
        if(lastNOut != 0.toLong()){
            if(layer is FeedForwardLayer)
                if(layer.nIn == 0.toLong()) layer.nIn = lastNOut else ;
            else if(layer is LocallyConnected1D)
                if(layer.nIn == 0.toLong()) layer.nIn = lastNOut else ;
            else if(layer is LocallyConnected2D)
                if(layer.nIn == 0.toLong()) layer.nIn = lastNOut else ;
        }

        if(layer is FeedForwardLayer)
            lastNOut = layer.nOut
        else if(layer is LocallyConnected1D)
            lastNOut = layer.nOut
        else if(layer is LocallyConnected2D)
            lastNOut = layer.nOut
        else if(layer is Bidirectional)
            lastNOut = layer.nOut
        */
        if (index != null)
            layer(index, layer)
        else
            layer(layer)
    }

    @NNConfDSL
    operator fun <L : Layer> L.unaryPlus(): L{
        addLayer(this)
        return this
    }

    @NNConfDSL
    operator fun <L : Layer.Builder<L>> L.unaryPlus(): Layer {
        val l = this.build<Layer>()
        +l
        return l
    }

    @NNConfDSL
    inline infix fun <L : Layer> L.addWith(builder: L.() -> Unit) = +apply(builder)

    @NNConfDSL
    inline infix fun <L : Layer.Builder<L>> L.addWith(builder: L.() -> Unit) = +apply(builder)

    @NNConfDSL
    infix fun <L : Layer> Int.layerAt(layer: L): L{
        addLayer(layer, this)
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


    @NNConfDSL
    fun globalConfig(builder: NeuralNetConfiguration.Builder.() -> Unit) {
        globalConfig.apply(builder)
    }

}
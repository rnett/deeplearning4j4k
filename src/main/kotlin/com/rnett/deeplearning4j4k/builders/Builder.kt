package com.rnett.deeplearning4j4k.builders

import com.rnett.deeplearning4j4k.NNConfDSL
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

@NNConfDSL
fun <T> neuralNetConfigutation(builder: NeuralNetConfiguration.Builder.() -> T) =
    NeuralNetConfiguration.Builder().run(builder)

@NNConfDSL
fun multiLayerNetworkConfig(builder: LayersBuilder.() -> Unit): LayersBuilder = neuralNetConfigutation {
    layers(builder)
}

@NNConfDSL
fun NeuralNetConfiguration.Builder.layers(builder: LayersBuilder.() -> Unit = {}) =
    LayersBuilder(this).apply(builder)

class Builder(newConf: NeuralNetConfiguration?) : NeuralNetConfiguration.Builder(newConf){
    constructor() : this(null)
}

fun MultiLayerConfiguration.multiLayerNetwork() = MultiLayerNetwork(this)

operator fun <L : Layer.Builder<L>> L.invoke(builder: L.() -> Unit) = apply(builder)
operator fun <L : NeuralNetConfiguration.Builder> L.invoke(builder: L.() -> Unit) = apply(builder)
operator fun <L : NeuralNetConfiguration.ListBuilder> L.invoke(builder: L.() -> Unit) = apply(builder)
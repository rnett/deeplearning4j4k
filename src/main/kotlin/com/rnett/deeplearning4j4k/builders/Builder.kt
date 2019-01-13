package com.rnett.deeplearning4j4k.builders

import com.rnett.deeplearning4j4k.NNConfDSL
import org.deeplearning4j.nn.conf.NeuralNetConfiguration

@NNConfDSL
fun neuralNetConfigutation(builder: NeuralNetConfiguration.Builder.() -> Unit) = NeuralNetConfiguration.Builder().apply(builder)

@NNConfDSL
fun NeuralNetConfiguration.Builder.layers(builder: LayersBuilder.() -> Unit) =
    LayersBuilder(this).apply(builder)

class Builder(newConf: NeuralNetConfiguration?) : NeuralNetConfiguration.Builder(newConf){
    constructor() : this(null)
}

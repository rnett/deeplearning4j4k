package com.rnett.deeplearning4j4k

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.BaseLayer
import org.nd4j.linalg.activations.Activation

var BaseLayer.activation
    get() = Activation.values().find { it.activationFunction.toString() == activationFn.toString() }
    set(value){
        activationFn = value?.activationFunction ?: Activation.IDENTITY.activationFunction
    }

var NeuralNetConfiguration.Builder.activation
    get() = Activation.values().find { it.activationFunction.toString() == activationFn.toString() }
    set(value){
        activationFn = value?.activationFunction ?: Activation.IDENTITY.activationFunction
    }

@DslMarker
annotation class NNConfDSL
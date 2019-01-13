package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.layers.LocallyConnected1D
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D
import org.nd4j.linalg.activations.Activation

@NNConfDSL
inline fun LayersBuilder.locallyConnected1DLayer(builder: LocallyConnected1D.Builder.() -> Unit = {}) =
    LocallyConnected1D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.locallyConnected1DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    kernelSize: Int = 2,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    inputSize: Int? = null,
    convolutionMode: ConvolutionMode = ConvolutionMode.Same,
    hasBias: Boolean = true,
    activation: Activation? = null,
    builder: LocallyConnected1D.Builder.() -> Unit = {}
) =
    LocallyConnected1D.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        kernelSize(kernelSize)
        stride(stride)
        padding(padding)
        dilation(dilation)
        if (inputSize != null)
            setInputSize(inputSize)
        convolutionMode(convolutionMode)
        hasBias(hasBias)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.locallyConnected2DLayer(builder: LocallyConnected2D.Builder.() -> Unit = {}) =
    LocallyConnected2D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.locallyConnected2DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    kernelSize: Pair<Int, Int> = 2.squared,
    stride: Pair<Int, Int> = 1.squared,
    padding: Pair<Int, Int> = 0.squared,
    dilation: Pair<Int, Int> = 1.squared,
    inputSize: Int? = null,
    convolutionMode: ConvolutionMode = ConvolutionMode.Same,
    hasBias: Boolean = true,
    activation: Activation? = null,
    builder: LocallyConnected2D.Builder.() -> Unit = {}
) =
    LocallyConnected2D.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        kernelSize(*kernelSize.toList().toIntArray())
        stride(*stride.toList().toIntArray())
        padding(*padding.toList().toIntArray())
        dilation(*dilation.toList().toIntArray())
        if (inputSize != null)
            setInputSize(inputSize)
        convolutionMode(convolutionMode)
        hasBias(hasBias)
        activation(activation)
    } addWith builder
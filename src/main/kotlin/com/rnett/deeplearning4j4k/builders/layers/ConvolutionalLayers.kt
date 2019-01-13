package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.layers.*
import org.nd4j.linalg.activations.IActivation


inline infix fun <T, R, S> Pair<T, R>.with(s: S) = Triple(first, second, s)
inline val <T> T.cubed get() = this to this with this
inline val <T> T.squared get() = this to this

@NNConfDSL
inline fun LayersBuilder.convolution1DLayer(builder: Convolution1DLayer.Builder.() -> Unit = {}) =
    Convolution1DLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.convolution1DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    kernelSize: Int = 0,
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: Convolution1DLayer.Builder.() -> Unit = {}
) =
    Convolution1DLayer.Builder(kernelSize, stride, padding).apply {
        dilation(dilation)
        nIn(nIn)
        nOut(nOut)
        hasBias(hasBias)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.convolution2DLayer(builder: ConvolutionLayer.Builder.() -> Unit = {}) =
    ConvolutionLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.convolution2DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    kernelSize: Pair<Int, Int> = 5 to 5,
    stride: Pair<Int, Int> = 1 to 1,
    padding: Pair<Int, Int> = 0 to 0,
    dilation: Pair<Int, Int> = 1 to 1,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: ConvolutionLayer.Builder.() -> Unit = {}
) =
    ConvolutionLayer.Builder(
        kernelSize.toList().toIntArray(),
        stride.toList().toIntArray(),
        padding.toList().toIntArray()
    ).apply {
        nIn(nIn)
        nOut(nOut)
        dilation(*dilation.toList().toIntArray())
        hasBias(hasBias)
        activation(activation)
    } addWith builder



@NNConfDSL
inline fun LayersBuilder.convolution3DLayer(builder: Convolution3D.Builder.() -> Unit = {}) =
    Convolution3D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.convolution3DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    kernelSize: Triple<Int, Int, Int> = 2.cubed,
    stride: Triple<Int, Int, Int> = 1.cubed,
    padding: Triple<Int, Int, Int> = 0.cubed,
    dilation: Triple<Int, Int, Int> = 1.cubed,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: Convolution3D.Builder.() -> Unit = {}
) =
    Convolution3D.Builder(
        kernelSize.toList().toIntArray(),
        stride.toList().toIntArray(),
        padding.toList().toIntArray()
    ).apply {
        dilation(*dilation.toList().toIntArray())
        nIn(nIn)
        nOut(nOut)
        hasBias(hasBias)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.separableConvolution2DLayer(builder: SeparableConvolution2D.Builder.() -> Unit = {}) =
    SeparableConvolution2D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.separableConvolution2DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    depthMultiplier: Int = 1,
    kernelSize: Pair<Int, Int> = 5 to 5,
    stride: Pair<Int, Int> = 1 to 1,
    padding: Pair<Int, Int> = 0 to 0,
    dilation: Pair<Int, Int> = 1 to 1,
    pointWiseConstraints: List<LayerConstraint> = emptyList(),
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: SeparableConvolution2D.Builder.() -> Unit = {}
) =
    SeparableConvolution2D.Builder(
        kernelSize.toList().toIntArray(),
        stride.toList().toIntArray(),
        padding.toList().toIntArray()
    ).apply {
        depthMultiplier(depthMultiplier)
        constrainPointWise(*pointWiseConstraints.toTypedArray())
        nIn(nIn)
        nOut(nOut)
        dilation(*dilation.toList().toIntArray())
        hasBias(hasBias)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.depthwiseConvolution2DLayer(builder: DepthwiseConvolution2D.Builder.() -> Unit = {}) =
    DepthwiseConvolution2D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.depthwiseConvolution2DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    depthMultiplier: Int = 1,
    kernelSize: Pair<Int, Int> = 5 to 5,
    stride: Pair<Int, Int> = 1 to 1,
    padding: Pair<Int, Int> = 0 to 0,
    dilation: Pair<Int, Int> = 1 to 1,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: DepthwiseConvolution2D.Builder.() -> Unit = {}
) =
    DepthwiseConvolution2D.Builder(
        kernelSize.toList().toIntArray(),
        stride.toList().toIntArray(),
        padding.toList().toIntArray()
    ).apply {
        depthMultiplier(depthMultiplier)
        nIn(nIn)
        nOut(nOut)
        dilation(*dilation.toList().toIntArray())
        hasBias(hasBias)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.deconvolution2DLayer(builder: Deconvolution2D.Builder.() -> Unit = {}) =
    Deconvolution2D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.deconvolution2DLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    kernelSize: Pair<Int, Int> = 5 to 5,
    stride: Pair<Int, Int> = 1 to 1,
    padding: Pair<Int, Int> = 0 to 0,
    dilation: Pair<Int, Int> = 1 to 1,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: Deconvolution2D.Builder.() -> Unit = {}
) =
    Deconvolution2D.Builder(
        kernelSize.toList().toIntArray(),
        stride.toList().toIntArray(),
        padding.toList().toIntArray()
    ).apply {
        nIn(nIn)
        nOut(nOut)
        dilation(*dilation.toList().toIntArray())
        hasBias(hasBias)
        activation(activation)
    } addWith builder

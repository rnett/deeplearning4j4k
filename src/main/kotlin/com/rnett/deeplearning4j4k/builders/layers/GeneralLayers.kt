package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer
import org.deeplearning4j.nn.conf.layers.util.MaskLayer
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions

@NNConfDSL
inline fun LayersBuilder.activationLayer(builder: ActivationLayer.() -> Unit) =
    +ActivationLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.activationLayer(activation: Activation, builder: ActivationLayer.() -> Unit) =
    +ActivationLayer(activation).apply(builder)

@NNConfDSL
inline fun LayersBuilder.activationLayer(activation: IActivation, builder: ActivationLayer.() -> Unit) =
    +ActivationLayer(activation).apply(builder)


@NNConfDSL
inline fun LayersBuilder.denseLayer(builder: DenseLayer.() -> Unit) =
    +DenseLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.denseLayer(nIn: Long, nOut: Long, builder: DenseLayer.() -> Unit) =
    +DenseLayer().apply{
        this.nIn = nIn
        this.nOut = nOut
    }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.dropoutLayer(builder: DropoutLayer.() -> Unit) =
    +DropoutLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.dropoutLayer(retainProbability: Double, builder: DropoutLayer.() -> Unit) =
    +DropoutLayer().apply{ this.iDropout = Dropout(retainProbability) }.apply(builder)

@NNConfDSL
inline fun LayersBuilder.dropoutLayer(dropout: IDropout, builder: DropoutLayer.() -> Unit) =
    +DropoutLayer().apply{ this.iDropout = dropout }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.embeddingLayer(builder: EmbeddingLayer.() -> Unit) =
    +EmbeddingLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.embeddingLayer(nIn: Long, nOut: Long, builder: EmbeddingLayer.() -> Unit) =
    +EmbeddingLayer().apply{
        this.nIn = nIn
        this.nOut = nOut
    }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.embeddingSequenceLayer(inferInputLength: Boolean = false, builder: EmbeddingSequenceLayer.() -> Unit) =
    +EmbeddingSequenceLayer().apply{
        this.isInferInputLength = inferInputLength
    }.apply(builder)

@NNConfDSL
inline fun LayersBuilder.embeddingSequenceLayer(inputLength: Int, builder: EmbeddingSequenceLayer.() -> Unit) =
    +EmbeddingSequenceLayer().apply{
        this.inputLength = inputLength
    }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.globalPoolingLayer(builder: GlobalPoolingLayer.() -> Unit) =
    +GlobalPoolingLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.globalPoolingLayer(
    poolingDimensions: List<Int>,
    poolingType: PoolingType,
    collapseDimensions: Boolean = true,
    pnorm: Int? = null,
    builder: GlobalPoolingLayer.() -> Unit) =
    +GlobalPoolingLayer().apply{
        this.poolingDimensions = poolingDimensions.toIntArray()
        this.poolingType = poolingType
        this.isCollapseDimensions = collapseDimensions

        if(pnorm != null)
            this.pnorm = pnorm
    }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.localResponseNormalization(builder: LocalResponseNormalization.() -> Unit) =
    +LocalResponseNormalization().apply(builder)

@NNConfDSL
inline fun LayersBuilder.localResponseNormalization(k: Double, n: Double, alpha: Double=0.0001, beta: Double = 0.4, builder: LocalResponseNormalization.() -> Unit) =
    +LocalResponseNormalization().apply(builder)


@NNConfDSL
inline fun LayersBuilder.locallyConnected1D(builder: LocallyConnected1D.() -> Unit) =
    +LocallyConnected1D.Builder().build().apply(builder)

@NNConfDSL
inline fun LayersBuilder.locallyConnected1D(
    nIn: Int, nOut: Int,
    activation: Activation = Activation.IDENTITY,
    kernelSize: Int,
    stride: Int,
    padding: Int?,
    convolutionMode: ConvolutionMode?,
    dilation: Int,
    inputSize: Int,
    builder: LocallyConnected1D.() -> Unit) =
    +LocallyConnected1D.Builder().apply{
        nIn(nIn)
        nOut(nOut)
        activation(activation)
        kernelSize(kernelSize)
        stride(stride)
        if(padding != null)
            padding(padding)

        if(convolutionMode != null)
            convolutionMode(convolutionMode)

        dilation(dilation)
        setInputSize(inputSize)

    }.build().apply(builder)


@NNConfDSL
inline fun LayersBuilder.locallyConnected2D(builder: LocallyConnected2D.() -> Unit) =
    +LocallyConnected2D.Builder().build().apply(builder)

@NNConfDSL
inline fun LayersBuilder.locallyConnected2D(
    nIn: Int, nOut: Int,
    activation: Activation = Activation.IDENTITY,
    kernelSize: Int,
    stride: Int,
    padding: Int?,
    convolutionMode: ConvolutionMode?,
    dilation: Int,
    inputSize: Int,
    builder: LocallyConnected2D.() -> Unit) =
    +LocallyConnected2D.Builder().apply{
        nIn(nIn)
        nOut(nOut)
        activation(activation)
        kernelSize(kernelSize)
        stride(stride)
        if(padding != null)
            padding(padding)

        if(convolutionMode != null)
            convolutionMode(convolutionMode)

        dilation(dilation)
        setInputSize(inputSize)

    }.build().apply(builder)


@NNConfDSL
inline fun LayersBuilder.lossLayer(builder: LossLayer.() -> Unit) =
    +LossLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.lossLayer(loss: ILossFunction, builder: LossLayer.() -> Unit) =
    +LossLayer().apply{this.lossFn = loss}.apply(builder)

@NNConfDSL
inline fun LayersBuilder.lossLayer(loss: LossFunctions.LossFunction, builder: LossLayer.() -> Unit) =
    +LossLayer().apply{this.lossFn = loss.iLossFunction}.apply(builder)


@NNConfDSL
inline fun LayersBuilder.outputLayer(builder: OutputLayer.() -> Unit) =
    +OutputLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.outputLayer(loss: ILossFunction, nIn: Long, nOut: Long, builder: OutputLayer.() -> Unit) =
    +OutputLayer().apply{
        this.lossFn = loss
        this.nIn = nIn
        this.nOut = nOut
    }.apply(builder)

@NNConfDSL
inline fun LayersBuilder.outputLayer(loss: LossFunctions.LossFunction, nIn: Long, nOut: Long, builder: OutputLayer.() -> Unit) =
    +OutputLayer().apply{
        this.lossFn = loss.iLossFunction
        this.nIn = nIn
        this.nOut = nOut
    }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.pooling1D(builder: Pooling1D.() -> Unit) =
    +Pooling1D().apply(builder)


@NNConfDSL
inline fun LayersBuilder.subsampling1DLayer(builder: Subsampling1DLayer.() -> Unit) =
    +Subsampling1DLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.subsampling1DLayer(
    poolingType: SubsamplingLayer.PoolingType,
    kernelSize: Int,
    stride: Int,
    padding: Int?,
    convolutionMode: ConvolutionMode?,
    pnorm: Int? = null,
    eps: Double = 1e-8,
    builder: Subsampling1DLayer.() -> Unit) =
    +Subsampling1DLayer.Builder().apply{
        kernelSize(kernelSize)
        stride(stride)
        poolingType(poolingType)

        eps(eps)

        if(padding != null)
            padding(padding)

        if(convolutionMode != null)
            convolutionMode(convolutionMode)

        if(pnorm != null)
            pnorm(pnorm)

    }.build().apply(builder)


@NNConfDSL
inline fun LayersBuilder.pooling2D(builder: Pooling2D.() -> Unit) =
    +Pooling2D().apply(builder)


@NNConfDSL
inline fun LayersBuilder.subsamplingLayer(builder: SubsamplingLayer.() -> Unit) =
    +SubsamplingLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.subsampling2DLayer(
    poolingType: SubsamplingLayer.PoolingType,
    kernelSize: Int,
    stride: Int,
    padding: Int?,
    convolutionMode: ConvolutionMode?,
    pnorm: Int? = null,
    eps: Double = 1e-8,
    builder: SubsamplingLayer.() -> Unit) =
    +SubsamplingLayer.Builder().apply{
        kernelSize(kernelSize)
        stride(stride)
        poolingType(poolingType)

        eps(eps)

        if(padding != null)
            padding(padding)

        if(convolutionMode != null)
            convolutionMode(convolutionMode)

        if(pnorm != null)
            pnorm(pnorm)

    }.build().apply(builder)


@NNConfDSL
inline fun LayersBuilder.upsampling1D(builder: Upsampling1D.() -> Unit) =
    +Upsampling1D().apply(builder)

@NNConfDSL
inline fun LayersBuilder.upsampling1D(size: Int, builder: Upsampling1D.() -> Unit) =
    +Upsampling1D().apply{ this.size = intArrayOf(size, size) /* its 2d under the hood, so set both sizes */ }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.upsampling2D(builder: Upsampling2D.() -> Unit) =
    +Upsampling2D().apply(builder)

//TODO check height and width order
@NNConfDSL
inline fun LayersBuilder.upsampling2D(height: Int, width: Int, builder: Upsampling2D.() -> Unit) =
    +Upsampling2D().apply{ this.size = intArrayOf(height, width) }.apply(builder)

@NNConfDSL
inline fun LayersBuilder.upsampling2D(allSizes: Int, builder: Upsampling2D.() -> Unit) =
    +Upsampling2D().apply{ this.size = intArrayOf(allSizes, allSizes) }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.upsampling3D(builder: Upsampling3D.() -> Unit) =
    +Upsampling3D().apply(builder)

//TODO check depth, width, and height order
@NNConfDSL
inline fun LayersBuilder.upsampling3D(depth: Int, width: Int, height: Int, builder: Upsampling3D.() -> Unit) =
    +Upsampling3D().apply{ this.size = intArrayOf(depth, width, height) }.apply(builder)

@NNConfDSL
inline fun LayersBuilder.upsampling3D(allSizes: Int, builder: Upsampling3D.() -> Unit) =
    +Upsampling3D().apply{ this.size = intArrayOf(allSizes, allSizes, allSizes) }.apply(builder)


@NNConfDSL
inline fun LayersBuilder.zeroPadding1DLayer(builder: ZeroPadding1DLayer.() -> Unit) =
    +ZeroPadding1DLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.zeroPadding1DLayer(paddingLeft: Int, paddingRight: Int, builder: ZeroPadding1DLayer.() -> Unit) =
    +ZeroPadding1DLayer(paddingLeft, paddingRight).apply(builder)


@NNConfDSL
inline fun LayersBuilder.zeroPaddingLayer(builder: ZeroPaddingLayer.() -> Unit) =
    +ZeroPaddingLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.zeroPaddingLayer(paddingHeight: Int, paddingWidth: Int, builder: ZeroPaddingLayer.() -> Unit) =
    +ZeroPaddingLayer(paddingHeight, paddingWidth).apply(builder)


@NNConfDSL
inline fun LayersBuilder.elementWiseMultiplicationLayer(builder: ElementWiseMultiplicationLayer.() -> Unit) =
    +ElementWiseMultiplicationLayer.Builder().build().apply(builder)

@NNConfDSL
inline fun LayersBuilder.elementWiseMultiplicationLayer(nIn: Long, nOut: Long, builder: ElementWiseMultiplicationLayer.() -> Unit) =
    +ElementWiseMultiplicationLayer.Builder().apply{
        nIn(nIn)
        nOut(nOut)
    }.build().apply(builder)


@NNConfDSL
inline fun LayersBuilder.maskLayer(builder: MaskLayer.() -> Unit) =
    +MaskLayer().apply(builder)


@NNConfDSL
inline fun LayersBuilder.maskZeroLayer(otherLayer: Layer, maskingValue: Double, builder: MaskZeroLayer.() -> Unit) =
    +MaskZeroLayer(otherLayer, maskingValue).apply(builder)
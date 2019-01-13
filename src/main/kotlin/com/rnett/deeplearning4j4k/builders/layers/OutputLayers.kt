package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.ocnn.OCNNOutputLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.lossfunctions.ILossFunction

@NNConfDSL
inline fun LayersBuilder.outputLayer(builder: OutputLayer.Builder.() -> Unit) =
    OutputLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.outputLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    lossFunction: ILossFunction? = null,
    hasBias: Boolean = true,
    builder: OutputLayer.Builder.() -> Unit
) =
    OutputLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        if (lossFunction != null)
            lossFunction(lossFunction)
        hasBias(hasBias)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.lossLayer(builder: LossLayer.Builder.() -> Unit) =
    LossLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.lossLayer(
    lossFunction: ILossFunction,
    hasBias: Boolean = true,
    builder: LossLayer.Builder.() -> Unit
) =
    LossLayer.Builder().apply {
        lossFunction(lossFunction)
        hasBias(hasBias)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.centerLossOutputLayer(builder: CenterLossOutputLayer.Builder.() -> Unit) =
    CenterLossOutputLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.centerLossOutputLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    alpha: Double = 0.25,
    lambda: Double = 2e-4,
    gradientCheck: Boolean = false,
    lossFunction: ILossFunction? = null,
    hasBias: Boolean = true,
    builder: CenterLossOutputLayer.Builder.() -> Unit
) =
    CenterLossOutputLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        alpha(alpha)
        lambda(lambda)
        gradientCheck(gradientCheck)
        if (lossFunction != null)
            lossFunction(lossFunction)
        hasBias(hasBias)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.cnn3DLossLayer(
    dataFormat: Convolution3D.DataFormat,
    builder: Cnn3DLossLayer.Builder.() -> Unit
) =
    Cnn3DLossLayer.Builder(dataFormat) addWith builder

@NNConfDSL
inline fun LayersBuilder.cnn3DLossLayer(
    dataFormat: Convolution3D.DataFormat,
    lossFunction: ILossFunction,
    nIn: Int = 0,
    nOut: Int = 0,
    hasBias: Boolean = true,
    builder: Cnn3DLossLayer.Builder.() -> Unit
) =
    Cnn3DLossLayer.Builder(dataFormat).apply {
        nIn(nIn)
        nOut(nOut)
        lossFunction(lossFunction)
        hasBias(hasBias)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.rnnLossLayer(builder: RnnLossLayer.Builder.() -> Unit) =
    RnnLossLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.rnnLossLayer(
    rnnLossFunction: ILossFunction,
    hasBias: Boolean = true,
    builder: RnnLossLayer.Builder.() -> Unit
) =
    RnnLossLayer.Builder().apply {
        lossFunction(rnnLossFunction)
        hasBias(hasBias)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.cnnLossLayer(builder: CnnLossLayer.Builder.() -> Unit) =
    CnnLossLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.cnnLossLayer(
    cnnLossFunction: ILossFunction,
    hasBias: Boolean = true,
    builder: CnnLossLayer.Builder.() -> Unit
) =
    CnnLossLayer.Builder().apply {
        lossFunction(cnnLossFunction)
        hasBias(hasBias)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.oCNNOutputLayer(builder: OCNNOutputLayer.Builder.() -> Unit) =
    OCNNOutputLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.oCNNOutputLayer(
    hiddenLayerSize: Int,
    nu: Double = 0.04,
    windowSize: Int = 10_000,
    activation: IActivation = Activation.IDENTITY.activationFunction,
    initialRValue: Double = 0.1,
    configureR: Boolean = true,
    nIn: Int = 0,
    nOut: Int = 0,
    lossFunction: ILossFunction? = null,
    hasBias: Boolean = true,
    builder: OCNNOutputLayer.Builder.() -> Unit
) =
    OCNNOutputLayer.Builder().apply {
        hiddenLayerSize(hiddenLayerSize)
        nu(nu)
        windowSize(windowSize)
        activation(activation)
        initialRValue(initialRValue)
        configureR(configureR)
        nIn(nIn)
        nOut(nOut)
        if (lossFunction != null)
            lossFunction(lossFunction)
        hasBias(hasBias)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.rnnOutputLayer(builder: RnnOutputLayer.Builder.() -> Unit) =
    RnnOutputLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.rnnOutputLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    lossFunction: ILossFunction? = null,
    hasBias: Boolean = true,
    builder: RnnOutputLayer.Builder.() -> Unit
) =
    RnnOutputLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        if (lossFunction != null)
            lossFunction(lossFunction)
        hasBias(hasBias)
    } addWith builder
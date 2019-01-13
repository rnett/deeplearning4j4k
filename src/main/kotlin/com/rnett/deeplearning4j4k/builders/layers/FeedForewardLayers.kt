package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation

@NNConfDSL
inline fun LayersBuilder.denseLayer(builder: DenseLayer.Builder.() -> Unit) =
    DenseLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.denseLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: DenseLayer.Builder.() -> Unit
) =
    DenseLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        hasBias(hasBias)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.denseLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    hasBias: Boolean = true,
    activation: Activation? = null,
    builder: DenseLayer.Builder.() -> Unit
) =
    DenseLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        hasBias(hasBias)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.dropoutLayer(builder: DropoutLayer.Builder.() -> Unit) =
    DropoutLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.dropoutLayer(
    dropout: Dropout,
    nIn: Int = 0,
    nOut: Int = 0,
    activation: IActivation? = null,
    builder: DropoutLayer.Builder.() -> Unit
) =
    DropoutLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        dropOut(dropout)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.dropoutLayer(
    dropout: IDropout,
    nIn: Int = 0,
    nOut: Int = 0,
    activation: IActivation? = null,
    builder: DropoutLayer.Builder.() -> Unit
) =
    DropoutLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        dropOut(dropout)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.dropoutLayer(
    dropout: Double,
    nIn: Int = 0,
    nOut: Int = 0,
    activation: IActivation? = null,
    builder: DropoutLayer.Builder.() -> Unit
) =
    DropoutLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        dropOut(dropout)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.embeddingLayer(builder: EmbeddingLayer.Builder.() -> Unit) =
    EmbeddingLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.embeddingLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: EmbeddingLayer.Builder.() -> Unit
) =
    EmbeddingLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        hasBias(hasBias)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.embeddingLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    hasBias: Boolean = true,
    activation: Activation? = null,
    builder: EmbeddingLayer.Builder.() -> Unit
) =
    EmbeddingLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        hasBias(hasBias)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.embeddingSequenceLayer(builder: EmbeddingSequenceLayer.Builder.() -> Unit) =
    EmbeddingSequenceLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.embeddingSequenceLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    inputLength: Int = 1,
    inferInputLength: Boolean = true,
    hasBias: Boolean = true,
    activation: IActivation? = null,
    builder: EmbeddingSequenceLayer.Builder.() -> Unit
) =
    EmbeddingSequenceLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        inputLength(inputLength)
        inferInputLength(inferInputLength)
        hasBias(hasBias)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.embeddingSequenceLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    inputLength: Int = 1,
    inferInputLength: Boolean = true,
    hasBias: Boolean = true,
    activation: Activation? = null,
    builder: EmbeddingSequenceLayer.Builder.() -> Unit
) =
    EmbeddingSequenceLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        inputLength(inputLength)
        inferInputLength(inferInputLength)
        hasBias(hasBias)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.pReLULayer(builder: PReLULayer.Builder.() -> Unit) =
    PReLULayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.pReLULayer(
    inputShape: List<Long>,
    sharedAxes: List<Long>,
    nIn: Int = 0,
    nOut: Int = 0,
    activation: IActivation? = null,
    builder: PReLULayer.Builder.() -> Unit
) =
    PReLULayer.Builder().apply {
        inputShape(*inputShape.toLongArray())
        sharedAxes(*sharedAxes.toLongArray())
        nIn(nIn)
        nOut(nOut)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.pReLULayer(
    inputShape: List<Long>,
    sharedAxes: List<Long>,
    nIn: Int = 0,
    nOut: Int = 0,
    activation: Activation? = null,
    builder: PReLULayer.Builder.() -> Unit
) =
    PReLULayer.Builder().apply {
        inputShape(*inputShape.toLongArray())
        sharedAxes(*sharedAxes.toLongArray())
        nIn(nIn)
        nOut(nOut)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.batchNormalizationLayer(builder: BatchNormalization.Builder.() -> Unit) =
    BatchNormalization.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.batchNormalizationLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    decay: Double = 0.9,
    eps: Double = 1e-5,
    isMinibatch: Boolean = true,
    lockGammaBeta: Boolean = false,
    gamma: Double = 1.0,
    beta: Double = 0.0,
    betaConstraints: List<LayerConstraint> = emptyList(),
    gammaConstraints: List<LayerConstraint> = emptyList(),
    cudnnAllowFallback: Boolean = true,
    activation: IActivation? = null,
    builder: BatchNormalization.Builder.() -> Unit
) =
    BatchNormalization.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        decay(decay)
        eps(eps)
        minibatch(isMinibatch)
        lockGammaBeta(lockGammaBeta)
        gamma(gamma)
        beta(beta)
        constrainBeta(*betaConstraints.toTypedArray())
        constrainGamma(*gammaConstraints.toTypedArray())
        cudnnAllowFallback(cudnnAllowFallback)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.batchNormalizationLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    decay: Double = 0.9,
    eps: Double = 1e-5,
    isMinibatch: Boolean = true,
    lockGammaBeta: Boolean = false,
    gamma: Double = 1.0,
    beta: Double = 0.0,
    betaConstraints: List<LayerConstraint> = emptyList(),
    gammaConstraints: List<LayerConstraint> = emptyList(),
    cudnnAllowFallback: Boolean = true,
    activation: Activation? = null,
    builder: BatchNormalization.Builder.() -> Unit
) =
    BatchNormalization.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        decay(decay)
        eps(eps)
        minibatch(isMinibatch)
        lockGammaBeta(lockGammaBeta)
        gamma(gamma)
        beta(beta)
        constrainBeta(*betaConstraints.toTypedArray())
        constrainGamma(*gammaConstraints.toTypedArray())
        cudnnAllowFallback(cudnnAllowFallback)
        activation(activation)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.elementWiseMultiplicationLayer(builder: ElementWiseMultiplicationLayer.Builder.() -> Unit) =
    ElementWiseMultiplicationLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.elementWiseMultiplicationLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    activation: IActivation? = null,
    builder: ElementWiseMultiplicationLayer.Builder.() -> Unit
) =
    ElementWiseMultiplicationLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.elementWiseMultiplicationLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    activation: Activation? = null,
    builder: ElementWiseMultiplicationLayer.Builder.() -> Unit
) =
    ElementWiseMultiplicationLayer.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        activation(activation)
    } addWith builder


//TODO repeat vector.  builder is bugged, takes a type arg, makes instantiation impossible


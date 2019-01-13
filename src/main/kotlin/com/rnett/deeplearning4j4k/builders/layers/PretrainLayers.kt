package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.conf.layers.AutoEncoder
import org.deeplearning4j.nn.conf.layers.variational.GaussianReconstructionDistribution
import org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.activations.impl.ActivationIdentity
import org.nd4j.linalg.lossfunctions.LossFunctions

@NNConfDSL
inline fun LayersBuilder.variationalAutoencoderLayer(builder: VariationalAutoencoder.Builder.() -> Unit = {}) =
    VariationalAutoencoder.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.variationalAutoencoderLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    encoderLayerSizes: List<Int> = listOf(100),
    decoderLayerSizes: List<Int> = listOf(100),
    reconstructionDistribution: ReconstructionDistribution = GaussianReconstructionDistribution(Activation.TANH),
    pzxActivation: IActivation = ActivationIdentity(),
    numSamples: Int = 1,
    lossFunction: LossFunctions.LossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY,
    visibleBiasInit: Double = 0.0,
    activation: IActivation? = null,
    builder: VariationalAutoencoder.Builder.() -> Unit = {}
) =
    VariationalAutoencoder.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        encoderLayerSizes(*encoderLayerSizes.toIntArray())
        decoderLayerSizes(*decoderLayerSizes.toIntArray())
        reconstructionDistribution(reconstructionDistribution)
        pzxActivationFn(pzxActivation)
        numSamples(numSamples)
        lossFunction(lossFunction)
        visibleBiasInit(visibleBiasInit)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.variationalAutoencoderLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    encoderLayerSizes: List<Int> = listOf(100),
    decoderLayerSizes: List<Int> = listOf(100),
    reconstructionLossFunction: LossFunctions.LossFunction,
    reconstructionActivationFunction: IActivation,
    pzxActivation: IActivation = ActivationIdentity(),
    numSamples: Int = 1,
    lossFunction: LossFunctions.LossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY,
    visibleBiasInit: Double = 0.0,
    activation: IActivation? = null,
    builder: VariationalAutoencoder.Builder.() -> Unit = {}
) =
    VariationalAutoencoder.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        encoderLayerSizes(*encoderLayerSizes.toIntArray())
        decoderLayerSizes(*decoderLayerSizes.toIntArray())
        lossFunction(lossFunction)
        lossFunction(reconstructionActivationFunction, reconstructionLossFunction)
        pzxActivationFn(pzxActivation)
        numSamples(numSamples)
        visibleBiasInit(visibleBiasInit)
        activation(activation)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.autoEncoderLayer(builder: AutoEncoder.Builder.() -> Unit = {}) =
    AutoEncoder.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.autoEncoderLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    corruptionLevel: Double = 3e-1,
    sparsity: Double = 0.0,
    lossFunction: LossFunctions.LossFunction = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY,
    visibleBiasInit: Double = 0.0,
    activation: IActivation? = null,
    builder: AutoEncoder.Builder.() -> Unit = {}
) =
    AutoEncoder.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        corruptionLevel(corruptionLevel)
        sparsity(sparsity)
        lossFunction(lossFunction)
        visibleBiasInit(visibleBiasInit)
        activation(activation)
    } addWith builder

package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.activations.impl.ActivationSigmoid

@NNConfDSL
inline fun LayersBuilder.simpleRnnLayer(builder: SimpleRnn.Builder.() -> Unit = {}) =
    SimpleRnn.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.simpleRnnLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    activation: IActivation? = null,
    recurrentConstraints: List<LayerConstraint> = emptyList(),
    inputWeightConstraints: List<LayerConstraint> = emptyList(),
    weightInitRecurrent: WeightInit? = null,
    distRecurrent: Distribution? = null,
    builder: SimpleRnn.Builder.() -> Unit = {}
) =
    SimpleRnn.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        activation(activation)
        constrainRecurrent(*recurrentConstraints.toTypedArray())
        constrainInputWeights(*inputWeightConstraints.toTypedArray())
        weightInitRecurrent(weightInitRecurrent)
        if (distRecurrent != null)
            weightInitRecurrent(distRecurrent)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.lSTMLayer(builder: LSTM.Builder.() -> Unit = {}) =
    LSTM.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.lSTMLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    forgetGateBiasInit: Double = 0.0,
    gateActivation: IActivation = ActivationSigmoid(),
    activation: IActivation? = null,
    recurrentConstraints: List<LayerConstraint> = emptyList(),
    inputWeightConstraints: List<LayerConstraint> = emptyList(),
    weightInitRecurrent: WeightInit? = null,
    distRecurrent: Distribution? = null,
    builder: LSTM.Builder.() -> Unit = {}
) =
    LSTM.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        forgetGateBiasInit(forgetGateBiasInit)
        gateActivationFunction(gateActivation)
        activation(activation)
        constrainRecurrent(*recurrentConstraints.toTypedArray())
        constrainInputWeights(*inputWeightConstraints.toTypedArray())
        weightInitRecurrent(weightInitRecurrent)
        if (distRecurrent != null)
            weightInitRecurrent(distRecurrent)
    } addWith builder

@NNConfDSL
inline fun LayersBuilder.lSTMLayer(
    nIn: Int = 0,
    nOut: Int = 0,
    forgetGateBiasInit: Double = 0.0,
    gateActivation: Activation = Activation.SIGMOID,
    activation: IActivation? = null,
    recurrentConstraints: List<LayerConstraint> = emptyList(),
    inputWeightConstraints: List<LayerConstraint> = emptyList(),
    weightInitRecurrent: WeightInit? = null,
    distRecurrent: Distribution? = null,
    builder: LSTM.Builder.() -> Unit = {}
) =
    LSTM.Builder().apply {
        nIn(nIn)
        nOut(nOut)
        forgetGateBiasInit(forgetGateBiasInit)
        gateActivationFunction(gateActivation)
        activation(activation)
        constrainRecurrent(*recurrentConstraints.toTypedArray())
        constrainInputWeights(*inputWeightConstraints.toTypedArray())
        weightInitRecurrent(weightInitRecurrent)
        if (distRecurrent != null)
            weightInitRecurrent(distRecurrent)
    } addWith builder

package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.builders.byVar
import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.dropout.Dropout
import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.layers.LossLayer
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions

interface ILossLayerBuilder : IBaseOutputLayerBuilder {

}

class LossLayerBuilder : LossLayer.Builder(), ILossLayerBuilder {
    override var loss
        get() = LossFunctions.LossFunction.values().find {
            super.lossFn.toString().toLowerCase().startsWith(it.name.toLowerCase())
        }
        set(v) {
            super.lossFunction(loss)
        }

    override var lossFunction: ILossFunction
        get() = super.lossFn
        set(value) {
            lossFunction(value)
        }

    private var _hasBias = true

    override var hasBias
        get() = _hasBias
        set(value) {
            _hasBias = value
            hasBias(value)
        }

    override var name: String?
        get() = super.layerName
        set(value) {
            name(value)
        }

    override var weightConstraints: List<LayerConstraint>
        get() = super.weightConstraints ?: listOf()
        set(value) {
            constrainWeights(*value.toTypedArray())
        }

    override var biasConstraints: List<LayerConstraint>
        get() = super.biasConstraints ?: listOf()
        set(value) {
            constrainBias(*value.toTypedArray())
        }

    override var biasUpdater: IUpdater?
        get() = super.biasUpdater
        set(value) {
            biasUpdater(value)
        }

    override var gradientNormalization: GradientNormalization?
        get() = super.gradientNormalization
        set(value) {
            gradientNormalization(value)
        }

    override var gradientNormalizationThreshold: Double
        get() = super.gradientNormalizationThreshold
        set(value) {
            gradientNormalizationThreshold(value)
        }

    override var weightNoise: IWeightNoise?
        get() = super.weightNoise
        set(value) {
            weightNoise(value)
        }

    override var nIn: Int
        get() {
            throw UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.")
        }
        set(v) {
            throw UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.")
        }

    override var nOut: Int
        get() {
            throw UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.")
        }
        set(v) {
            throw UnsupportedOperationException("Ths layer has no parameters, thus nIn will always equal nOut.")
        }

    override var activation
        get() = Activation.values().find { it.activationFunction.toString() == _activation.toString() }
        set(v) {
            _activation = v?.activationFunction
        }

    override var _activation: IActivation? by byVar(::activationFn)

    override var weightInit: Distribution?
        get() = super.dist
        set(v) {
            weightInit(v)
        }

    override var _weightInit: WeightInit?
        get() = super.weightInit
        set(v) {
            super.weightInit(v)
        }

    override var biasInit: Double?
        get() = super.biasInit
        set(v) {
            super.biasInit(v ?: Double.NaN)
        }

    var distribution: Distribution?
        get() = super.dist
        set(v) {
            super.dist(v)
        }

    override var dropout: Double?
        get() = (super.iDropout as? Dropout)?.p
        set(v) {
            dropOut(v ?: 0.0)
        }

    override var _dropout: IDropout?
        get() = super.iDropout
        set(v) {
            super.dropOut(v)
        }

    override var allParamConstraints: List<LayerConstraint>
        get() = super.allParamConstraints ?: listOf()
        set(v) {
            super.allParamConstraints = v
        }

    override var updater: IUpdater?
        get() = super.iupdater
        set(v) {
            super.iupdater = v
        }

    override var l1
        get() = super.l1
        set(v) {
            super.l1 = v
        }

    override var l1Bias
        get() = super.l1Bias
        set(v) {
            super.l1Bias = v
        }

    override var l2
        get() = super.l2
        set(v) {
            super.l2 = v
        }

    override var l2Bias
        get() = super.l2Bias
        set(v) {
            super.l2Bias = v
        }
}

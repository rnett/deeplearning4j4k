package com.rnett.deeplearning4j4k.builders.layers

import org.deeplearning4j.nn.api.layers.LayerConstraint
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.dropout.IDropout
import org.deeplearning4j.nn.conf.weightnoise.IWeightNoise
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.IActivation
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.lossfunctions.ILossFunction
import org.nd4j.linalg.lossfunctions.LossFunctions

interface ILayerBuilder {
    var name: String?
    var dropout: Double?
    var _dropout: IDropout?
    var allParamConstraints: List<LayerConstraint>
    var weightConstraints: List<LayerConstraint>
    var biasConstraints: List<LayerConstraint>
}

interface IBaseLayerBuilder : ILayerBuilder {
    var activation: Activation?
    var _activation: IActivation?

    var weightInit: Distribution?
    var _weightInit: WeightInit?
    var biasInit: Double?

    var l1: Double
    var l2: Double
    var l1Bias: Double
    var l2Bias: Double

    var updater: IUpdater?

    var biasUpdater: IUpdater?

    var gradientNormalization: GradientNormalization?
    var gradientNormalizationThreshold: Double

    var weightNoise: IWeightNoise?
}

interface IFeedForewardLayerBuilder : IBaseLayerBuilder {
    var nIn: Int
    var nOut: Int
}

interface IBaseOutputLayerBuilder : IFeedForewardLayerBuilder {
    var hasBias: Boolean
    var loss: LossFunctions.LossFunction?
    var lossFunction: ILossFunction
}


/*
TODO
    working from https://deeplearning4j.org/docs/latest/deeplearning4j-cheat-sheet#layers-ff
    currently on RnnOutputLayer
 */


package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.conf.layers.LocalResponseNormalization
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer
import org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer

@NNConfDSL
inline fun LayersBuilder.yolo2OutputLayer(builder: Yolo2OutputLayer.Builder.() -> Unit = {}) =
    Yolo2OutputLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.maskZeroLayer(builder: MaskZeroLayer.Builder.() -> Unit = {}) =
    MaskZeroLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.localResponseNormalizationLayer(builder: LocalResponseNormalization.Builder.() -> Unit = {}) =
    LocalResponseNormalization.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.localResponseNormalizationLayer(
    k: Double = 2.0,
    n: Double = 5.0,
    alpha: Double = 1e-4,
    beta: Double = 0.75,
    builder: LocalResponseNormalization.Builder.() -> Unit = {}
) =
    LocalResponseNormalization.Builder(k, alpha, beta).apply {
        n(n)
    } addWith builder

//TODO space to batch layer, can't instantiate builder


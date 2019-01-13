package com.rnett.deeplearning4j4k.builders.layers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.layers.*
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping3D

@NNConfDSL
inline fun LayersBuilder.zeroPadding1DLayer(builder: ZeroPadding1DLayer.Builder.() -> Unit = {}) =
    ZeroPadding1DLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding1DLayer(
    paddingLeft: Int, paddingRight: Int,
    builder: ZeroPadding1DLayer.Builder.() -> Unit = {}
) =
    ZeroPadding1DLayer.Builder(paddingLeft, paddingRight) addWith builder


@NNConfDSL
inline fun LayersBuilder.zeroPadding2DLayer(builder: ZeroPaddingLayer.Builder.() -> Unit = {}) =
    ZeroPaddingLayer.Builder(0, 0) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding2DLayer(
    paddingTop: Int, paddingBottom: Int,
    paddingLeft: Int, paddingRight: Int,
    builder: ZeroPaddingLayer.Builder.() -> Unit = {}
) =
    ZeroPaddingLayer.Builder(paddingTop, paddingBottom, paddingLeft, paddingRight) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding2DLayer(
    paddingHeight: Int, paddingWidth: Int,
    builder: ZeroPaddingLayer.Builder.() -> Unit = {}
) =
    ZeroPaddingLayer.Builder(paddingHeight, paddingWidth) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding2DLayer(
    paddingAll: Int,
    builder: ZeroPaddingLayer.Builder.() -> Unit = {}
) =
    ZeroPaddingLayer.Builder(paddingAll, paddingAll) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding2DLayer(
    padding: List<Int>,
    builder: ZeroPaddingLayer.Builder.() -> Unit = {}
) =
    ZeroPaddingLayer.Builder(padding.toIntArray()) addWith builder


@NNConfDSL
inline fun LayersBuilder.zeroPadding3DLayer(builder: ZeroPadding3DLayer.Builder.() -> Unit = {}) =
    ZeroPadding3DLayer.Builder(0) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding3DLayer(
    paddingLeftD: Int, paddingRightD: Int,
    paddingLeftH: Int, paddingRightH: Int,
    paddingLeftW: Int, paddingRightW: Int,
    builder: ZeroPadding3DLayer.Builder.() -> Unit = {}
) =
    ZeroPadding3DLayer.Builder(
        paddingLeftD,
        paddingRightD,
        paddingLeftH,
        paddingRightH,
        paddingLeftW,
        paddingRightW
    ) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding3DLayer(
    paddingDepth: Int, paddingHeight: Int, paddingWidth: Int,
    builder: ZeroPadding3DLayer.Builder.() -> Unit = {}
) =
    ZeroPadding3DLayer.Builder(paddingDepth, paddingHeight, paddingWidth) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding3DLayer(
    paddingAll: Int,
    builder: ZeroPadding3DLayer.Builder.() -> Unit = {}
) =
    ZeroPadding3DLayer.Builder(paddingAll) addWith builder

@NNConfDSL
inline fun LayersBuilder.zeroPadding3DLayer(
    padding: List<Int>,
    builder: ZeroPadding3DLayer.Builder.() -> Unit = {}
) =
    ZeroPadding3DLayer.Builder(padding.toIntArray()) addWith builder


@NNConfDSL
inline fun LayersBuilder.cropping1DLayer(builder: Cropping1D.Builder.() -> Unit = {}) =
    Cropping1D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping1DLayer(croppingTopBottom: Int, builder: Cropping1D.Builder.() -> Unit = {}) =
    Cropping1D.Builder(croppingTopBottom) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping1DLayer(
    croppingTop: Int, croppingBottom: Int,
    builder: Cropping1D.Builder.() -> Unit = {}
) =
    Cropping1D.Builder(croppingTop, croppingBottom) addWith builder


@NNConfDSL
inline fun LayersBuilder.cropping2DLayer(builder: Cropping2D.Builder.() -> Unit = {}) =
    Cropping2D.Builder(0, 0) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping2DLayer(
    croppingTop: Int, croppingBottom: Int,
    croppingLeft: Int, croppingRight: Int,
    builder: Cropping2D.Builder.() -> Unit = {}
) =
    Cropping2D.Builder(croppingTop, croppingBottom, croppingLeft, croppingRight) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping2DLayer(
    croppingHeight: Int, croppingWidth: Int,
    builder: Cropping2D.Builder.() -> Unit = {}
) =
    Cropping2D.Builder(croppingHeight, croppingWidth) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping2DLayer(
    croppingAll: Int,
    builder: Cropping2D.Builder.() -> Unit = {}
) =
    Cropping2D.Builder(croppingAll, croppingAll) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping2DLayer(
    cropping: List<Int>,
    builder: Cropping2D.Builder.() -> Unit = {}
) =
    Cropping2D.Builder(cropping.toIntArray()) addWith builder


@NNConfDSL
inline fun LayersBuilder.cropping3DLayer(builder: Cropping3D.Builder.() -> Unit = {}) =
    Cropping3D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping3DLayer(
    cropLeftD: Int, cropRightD: Int,
    cropLeftH: Int, cropRightH: Int,
    cropLeftW: Int, cropRightW: Int,
    builder: Cropping3D.Builder.() -> Unit = {}
) =
    Cropping3D.Builder(cropLeftD, cropRightD, cropLeftH, cropRightH, cropLeftW, cropRightW) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping3DLayer(
    croppingDepth: Int, croppingHeight: Int, croppingWidth: Int,
    builder: Cropping3D.Builder.() -> Unit = {}
) =
    Cropping3D.Builder(croppingDepth, croppingHeight, croppingWidth) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping3DLayer(
    croppingAll: Int,
    builder: Cropping3D.Builder.() -> Unit = {}
) =
    Cropping3D.Builder(croppingAll, croppingAll, croppingAll) addWith builder

@NNConfDSL
inline fun LayersBuilder.cropping3DLayer(
    cropping: List<Int>,
    builder: Cropping3D.Builder.() -> Unit = {}
) =
    Cropping3D.Builder(cropping.toIntArray()) addWith builder


@NNConfDSL
inline fun LayersBuilder.upsampling1DLayer(builder: Upsampling1D.Builder.() -> Unit = {}) =
    Upsampling1D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.upsampling1DLayer(size: Int, builder: Upsampling1D.Builder.() -> Unit = {}) =
    Upsampling1D.Builder(size) addWith builder


@NNConfDSL
inline fun LayersBuilder.upsampling2DLayer(builder: Upsampling2D.Builder.() -> Unit = {}) =
    Upsampling2D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.upsampling2DLayer(allSizes: Int, builder: Upsampling2D.Builder.() -> Unit = {}) =
    Upsampling2D.Builder(allSizes) addWith builder

@NNConfDSL
inline fun LayersBuilder.upsampling2DLayer(height: Int, width: Int, builder: Upsampling2D.Builder.() -> Unit = {}) =
    upsampling2DLayer(listOf(height, width), builder)

@NNConfDSL
inline fun LayersBuilder.upsampling2DLayer(sizes: List<Int>, builder: Upsampling2D.Builder.() -> Unit = {}) =
    Upsampling2D.Builder() addWith { size(sizes.toIntArray()); builder() }


@NNConfDSL
inline fun LayersBuilder.upsampling3DLayer(builder: Upsampling3D.Builder.() -> Unit = {}) =
    Upsampling3D.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.upsampling3DLayer(allSizes: Int, builder: Upsampling3D.Builder.() -> Unit = {}) =
    Upsampling3D.Builder(allSizes) addWith builder

@NNConfDSL
inline fun LayersBuilder.upsampling3DLayer(
    depth: Int,
    width: Int,
    height: Int,
    builder: Upsampling3D.Builder.() -> Unit = {}
) =
    upsampling3DLayer(listOf(depth, width, height), builder)

@NNConfDSL
inline fun LayersBuilder.upsampling3DLayer(sizes: List<Int>, builder: Upsampling3D.Builder.() -> Unit = {}) =
    Upsampling3D.Builder() addWith { size(sizes.toIntArray()); builder() }


@NNConfDSL
inline fun LayersBuilder.subsampling1DLayer(builder: Subsampling1DLayer.Builder.() -> Unit = {}) =
    Subsampling1DLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.subsampling1DLayer(
    poolingType: PoolingType,
    convolutionMode: ConvolutionMode? = null,
    kernelSize: Int = 2,
    stride: Int = 1,
    padding: Int = 0,
    builder: Subsampling1DLayer.Builder.() -> Unit = {}
) =
    Subsampling1DLayer.Builder(poolingType, kernelSize, stride, padding).apply {
        convolutionMode(convolutionMode)
    } addWith builder


@NNConfDSL
inline fun LayersBuilder.subsampling2DLayer(builder: SubsamplingLayer.Builder.() -> Unit = {}) =
    SubsamplingLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.subsampling2DLayer(
    poolingType: PoolingType,
    convolutionMode: ConvolutionMode? = null,
    kernelSize: Pair<Int, Int> = 2 to 2,
    stride: Pair<Int, Int> = 1 to 1,
    padding: Pair<Int, Int> = 0 to 0,
    dilation: Pair<Int, Int> = 1 to 1,
    builder: SubsamplingLayer.Builder.() -> Unit = {}
) =
    SubsamplingLayer.Builder(
        poolingType,
        kernelSize.toList().toIntArray(),
        stride.toList().toIntArray(),
        padding.toList().toIntArray()
    ) addWith {
        convolutionMode(convolutionMode)
        dilation(*dilation.toList().toIntArray())
        builder()
    }

@NNConfDSL
inline fun LayersBuilder.subsampling3DLayer(builder: Subsampling3DLayer.Builder.() -> Unit = {}) =
    Subsampling3DLayer.Builder() addWith builder

@NNConfDSL
inline fun LayersBuilder.subsampling3DLayer(
    poolingType: PoolingType,
    convolutionMode: ConvolutionMode = ConvolutionMode.Same,
    kernelSize: Triple<Int, Int, Int> = 1.cubed,
    stride: Triple<Int, Int, Int> = 2.cubed,
    padding: Triple<Int, Int, Int> = 0.cubed,
    dilation: Triple<Int, Int, Int> = 1.cubed,
    builder: Subsampling3DLayer.Builder.() -> Unit = {}
) =
    Subsampling3DLayer.Builder(
        poolingType,
        kernelSize.toList().toIntArray(),
        stride.toList().toIntArray(),
        padding.toList().toIntArray()
    ) addWith {
        convolutionMode(convolutionMode)
        dilation(dilation.first, dilation.second, dilation.third)
        builder()
    }

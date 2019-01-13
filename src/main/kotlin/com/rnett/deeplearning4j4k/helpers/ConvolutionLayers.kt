package com.rnett.deeplearning4j4k.helpers

import com.rnett.deeplearning4j4k.NNConfDSL
import com.rnett.deeplearning4j4k.builders.LayersBuilder
import org.deeplearning4j.nn.conf.ConvolutionMode
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer
import org.deeplearning4j.nn.conf.layers.Convolution3D
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer

@NNConfDSL
inline fun LayersBuilder.convolution1DLayer(builder: Convolution1DLayer.() -> Unit) =
    +Convolution1DLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.convolution1DLayer(
    kernelSize: Int,
    stride: Int,
    padding: Int?,
    dilation: List<Int>,
    convolutionMode: ConvolutionMode?,
    builder: Convolution1DLayer.() -> Unit) =
    +Convolution1DLayer.Builder().apply{
        kernelSize(kernelSize)
        stride(stride)
        dilation(*dilation.toIntArray())

        if(padding != null)
            padding(padding)

        if(convolutionMode != null)
            convolutionMode(convolutionMode)

    }.build().apply(builder)

@NNConfDSL
inline fun LayersBuilder.convolution2DLayer(builder: ConvolutionLayer.() -> Unit) =
    +ConvolutionLayer().apply(builder)

@NNConfDSL
inline fun LayersBuilder.convolution2DLayer(
    kernelSize: Pair<Int, Int>,
    stride: Pair<Int, Int>,
    padding: Pair<Int, Int>?,
    dilation: List<Int>,
    convolutionMode: ConvolutionMode?,
    builder: ConvolutionLayer.() -> Unit) =
    +ConvolutionLayer.Builder().apply{
        kernelSize(*kernelSize.toList().toIntArray())
        stride(*stride.toList().toIntArray())
        dilation(*dilation.toIntArray())

        if(padding != null)
            padding(*padding.toList().toIntArray())

        if(convolutionMode != null)
            convolutionMode(convolutionMode)

    }.build().apply(builder)

@NNConfDSL
inline fun LayersBuilder.convolution3DLayer(builder: Convolution3D.() -> Unit) =
    +Convolution3D().apply(builder)

@NNConfDSL
inline fun LayersBuilder.convolution3DLayer(
    kernelSize: Pair<Int, Int>,
    stride: Pair<Int, Int>,
    padding: Pair<Int, Int>?,
    dilation: List<Int>,
    convolutionMode: ConvolutionMode?,
    dataFormat: Convolution3D.DataFormat = Convolution3D.DataFormat.NCDHW,
    builder: Convolution3D.() -> Unit) =
    +Convolution3D.Builder().apply{
        kernelSize(*kernelSize.toList().toIntArray())
        stride(*stride.toList().toIntArray())
        dilation(*dilation.toIntArray())
        dataFormat(dataFormat)

        if(padding != null)
            padding(*padding.toList().toIntArray())

        if(convolutionMode != null)
            convolutionMode(convolutionMode)

    }.build().apply(builder)

//https://deeplearning4j.org/docs/latest/deeplearning4j-nn-convolutional
// start froim deconvolution2D
package com.rnett.deeplearning4j4k

import com.rnett.deeplearning4j4k.builders.layers
import com.rnett.deeplearning4j4k.builders.layers.denseLayer
import com.rnett.deeplearning4j4k.builders.neuralNetConfigutation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.ActivationLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation


fun main() {
    neuralNetConfigutation {

        weightInit = WeightInit.XAVIER
        activation =Activation.RELU
        optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT


        layers{

            ActivationLayer(Activation.LEAKYRELU) addWith {

            }

            denseLayer {
                activation = Activation.LEAKYRELU

            }


        }

    }
}
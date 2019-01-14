import com.rnett.deeplearning4j4k.builders.layers.denseLayer
import com.rnett.deeplearning4j4k.builders.multiLayerNetwork
import com.rnett.deeplearning4j4k.builders.multiLayerNetworkConfig
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation


fun main() {
    multiLayerNetworkConfig {

        globalConfig {
            weightInit = WeightInit.XAVIER
            activationFn = Activation.RELU.activationFunction
            optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
        }

        denseLayer(100, 10)

    }.build().multiLayerNetwork()
}
package khudyakov_udod.perceptron.network;

import khudyakov_udod.perceptron.functions.FunctionImpl;
import khudyakov_udod.perceptron.network.entities.ActiveLayer;
import khudyakov_udod.perceptron.network.entities.Layer;
import khudyakov_udod.perceptron.network.entities.Neuron;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class Net {
    private final Layer inputLayer;
    private final List<ActiveLayer> hiddenLayers;
    private final ActiveLayer outputLayer;

    private final float rate;
    private float avgE = 0;
    private int exampleCount = 0;

    public Net(Layer inputLayer, List<ActiveLayer> hiddenLayers, ActiveLayer outputLayer, float rate) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.rate = rate;
    }

    private void calcNeuronOutputs(FunctionImpl functionImpl, List<Neuron> neurons, List<Float> bias) {
        for (int j = 0, neuronsSize = neurons.size(); j < neuronsSize; ++j) {
            Neuron neuron = neurons.get(j);

            float output = 0;
            Set<Map.Entry<Neuron, Float>> connections = neuron.getConnections().entrySet();

            for (Map.Entry<Neuron, Float> entry : connections) {
                output += entry.getValue() * entry.getKey().getActivatedOutput();
            }

            output += bias.get(j);

            neuron.setPureOutput(output);
            neuron.setActivatedOutput(functionImpl.getFunction().applyFunc(output, functionImpl.getA()));
        }
    }

    private void calcHiddenLayerOutputs() {
        for (ActiveLayer hiddenLayer : hiddenLayers) {
            calcNeuronOutputs(hiddenLayer.getFunctionImpl(), hiddenLayer.getNeurons(), hiddenLayer.getBias());
        }
    }

    private void calcOutputLayerOutputs() {
        calcNeuronOutputs(outputLayer.getFunctionImpl(), outputLayer.getNeurons(), outputLayer.getBias());
    }

    private void correctWeights(Neuron neuron) {
        Set<Neuron> connectedNeurons = neuron.getConnections().keySet();

        for (Neuron connectedNeuron : connectedNeurons) {
            float weightCorrection = ((-rate) * connectedNeuron.getActivatedOutput() * neuron.getDelta());
            float prevWeight = neuron.getConnections().get(connectedNeuron);
            neuron.getConnections().put(connectedNeuron, prevWeight + weightCorrection);
        }
    }

    private float calcDerivative(FunctionImpl functionImpl, Neuron neuron) {
        if (functionImpl.isSimplifyAvailable()) {
            return functionImpl.getFunction().applySimplifiedDeriveFunc(neuron.getActivatedOutput(), functionImpl.getA());
        } else {
            return functionImpl.getFunction().applyDeriveFunc(neuron.getPureOutput(), functionImpl.getA());
        }
    }

    private void recalculateOutputLayerWeights(float[] outputData) {
        List<Neuron> outputNeurons = outputLayer.getNeurons();
        FunctionImpl functionImpl = outputLayer.getFunctionImpl();

        for (int i = 0, neuronsSize = outputNeurons.size(); i < neuronsSize; ++i) {
            Neuron neuron = outputNeurons.get(i);
            neuron.setDelta(calcDerivative(functionImpl, neuron) * (neuron.getActivatedOutput() - outputData[i]));
            correctWeights(neuron);
        }
    }

    private void recalculateHiddenLayerWeights() {
        for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
            ActiveLayer hiddenLayer = hiddenLayers.get(i);
            List<Neuron> neurons = hiddenLayer.getNeurons();
            FunctionImpl functionImpl = hiddenLayer.getFunctionImpl();

            for (Neuron neuron : neurons) {
                float derivative = calcDerivative(functionImpl, neuron);

                ActiveLayer nextLayer = (i == hiddenLayers.size() - 1) ? outputLayer : hiddenLayers.get(i + 1);

                float correction = 0;
                List<Neuron> nextLayerNeurons = nextLayer.getNeurons();

                for (Neuron nextLayerNeuron : nextLayerNeurons) {
                    Map<Neuron, Float> connections = nextLayerNeuron.getConnections();

                    if (connections.containsKey(neuron)) {
                        correction += nextLayerNeuron.getActivatedOutput() * connections.get(neuron);
                    }
                }

                neuron.setDelta(correction * derivative);
                correctWeights(neuron);
            }
        }
    }


    private float applyRateFunction(float[] targetOutput, float[] calcOutput) {
        float sum = 0;
        for (int i = 0; i < targetOutput.length; ++i) {
            float difference = targetOutput[i] - calcOutput[i];
            sum += (difference * difference);
        }
        return sum / 2;
    }

    private void calcError(float[] outputData) {
        float[] calcOutput = new float[outputLayer.getNeurons().size()];

        List<Neuron> neurons = outputLayer.getNeurons();
        for (int i = 0, neuronsSize = neurons.size(); i < neuronsSize; ++i) {
            calcOutput[i] = neurons.get(i).getActivatedOutput();
        }

        avgE += applyRateFunction(outputData, calcOutput);
        exampleCount += 1;
    }

    public void trainExample(float[] inputData, float[] outputData) {
        if (inputData.length != inputLayer.getNeurons().size()) {
            throw new IllegalStateException("Incorrect data dimension for input layer");
        } else if (outputData.length != outputLayer.getNeurons().size()) {
            throw new IllegalStateException("Incorrect data dimension for output layer");
        } else {

            List<Neuron> neurons = inputLayer.getNeurons();
            for (int i = 0, neuronsSize = neurons.size(); i < neuronsSize; ++i) {
                neurons.get(i).setActivatedOutput(inputData[i]);
            }

            calcHiddenLayerOutputs();
            calcOutputLayerOutputs();

            recalculateOutputLayerWeights(outputData);
            recalculateHiddenLayerWeights();

            calcError(outputData);
        }
    }

    public float showAverageError() {
        float E = avgE / exampleCount;

        exampleCount = 0;
        avgE = 0;

        return E;
    }
}

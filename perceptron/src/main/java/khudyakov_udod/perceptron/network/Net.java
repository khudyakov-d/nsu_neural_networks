package khudyakov_udod.perceptron.network;

import khudyakov_udod.perceptron.functions.Function;
import khudyakov_udod.perceptron.network.entities.layers.ActiveLayer;
import khudyakov_udod.perceptron.network.entities.layers.Layer;
import khudyakov_udod.perceptron.network.entities.neurons.ActiveNeuron;
import khudyakov_udod.perceptron.network.entities.neurons.InputNeuron;
import khudyakov_udod.perceptron.network.entities.neurons.Neuron;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class Net {
    private final Layer inputLayer;
    private final List<Layer> hiddenLayers;
    private final Layer outputLayer;

    private final float rate;
    private float avgE = 0;
    private int exampleCount = 0;

    Net(Layer inputLayer, List<Layer> hiddenLayers, Layer outputLayer, float rate) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;
        this.rate = rate;
    }

    private void calcFirstHiddenLayer() {
        ActiveLayer hiddenLayer = (ActiveLayer) hiddenLayers.get(0);

        List<Neuron> neurons = hiddenLayer.getNeurons();
        List<Float> bias = hiddenLayer.getBias();
        Function function = hiddenLayer.getFunctionImpl();

        for (int i = 0, neuronsSize = neurons.size(); i < neuronsSize; ++i) {
            ActiveNeuron neuron = (ActiveNeuron) neurons.get(i);

            float output = 0;
            Set<Map.Entry<Neuron, Float>> connections = neuron.getConnections().entrySet();

            for (Map.Entry<Neuron, Float> entry : connections) {
                Float neuronOutput = ((InputNeuron) entry.getKey()).getOutput();
                if (neuronOutput != null) {
                    output += entry.getValue() * neuronOutput;
                }
            }
            output += bias.get(i);

            neuron.setPureOutput(output);
            neuron.setActivatedOutput(function.applyFunc(output));
        }
    }

    private void calcNeuronOutputs(Function functionImpl, List<Neuron> neurons, List<Float> bias) {
        for (int i = 0, neuronsSize = neurons.size(); i < neuronsSize; ++i) {
            ActiveNeuron neuron = (ActiveNeuron) neurons.get(i);

            float output = 0;
            Set<Map.Entry<Neuron, Float>> connections = neuron.getConnections().entrySet();

            for (Map.Entry<Neuron, Float> entry : connections) {
                output += entry.getValue() * ((ActiveNeuron) entry.getKey()).getActivatedOutput();
            }

            output += bias.get(i);

            neuron.setPureOutput(output);
            neuron.setActivatedOutput(functionImpl.applyFunc(output));
        }
    }

    private void calcOtherHiddenLayers() {
        for (int i = 1, hiddenLayersSize = hiddenLayers.size(); i < hiddenLayersSize; i++) {
            ActiveLayer hiddenLayer = (ActiveLayer) hiddenLayers.get(i);
            calcNeuronOutputs(hiddenLayer.getFunctionImpl(), hiddenLayer.getNeurons(), hiddenLayer.getBias());
        }
    }

    private void calcOutputLayerOutputs() {
        ActiveLayer layer = (ActiveLayer) outputLayer;
        calcNeuronOutputs(layer.getFunctionImpl(), layer.getNeurons(), layer.getBias());
    }

    private void correctLastWeights(ActiveNeuron neuron) {
        Set<Neuron> connectedNeurons = neuron.getConnections().keySet();

        for (Neuron connectedNeuron : connectedNeurons) {
            InputNeuron inputNeuron = ((InputNeuron) connectedNeuron);

            if (inputNeuron.getOutput() != null) {
                float weightCorrection = (-rate) * inputNeuron.getOutput() * neuron.getDelta();
                float prevWeight = neuron.getConnections().get(inputNeuron);
                neuron.getConnections().put(connectedNeuron, prevWeight + weightCorrection);
            }
        }
    }

    private void correctWeights(ActiveNeuron neuron) {
        Set<Neuron> connectedNeurons = neuron.getConnections().keySet();

        for (Neuron connectedNeuron : connectedNeurons) {
            ActiveNeuron activeNeuron = ((ActiveNeuron) connectedNeuron);

            float weightCorrection = (-rate) * activeNeuron.getActivatedOutput() * neuron.getDelta();
            float prevWeight = neuron.getConnections().get(activeNeuron);

            neuron.getConnections().put(connectedNeuron, prevWeight + weightCorrection);
        }
    }

    private float calcDerivative(Function functionImpl, ActiveNeuron neuron) {
        if (functionImpl.isSimplifyAvailable()) {
            return functionImpl.applySimplifiedDerivativeFunc(neuron.getActivatedOutput());
        } else {
            return functionImpl.applyDerivativeFunc(neuron.getPureOutput());
        }
    }

    private void recalculateOutputLayerWeights(float[] outputData) {
        ActiveLayer layer = (ActiveLayer) outputLayer;
        List<Neuron> outputHNeurons = layer.getNeurons();
        Function functionImpl = layer.getFunctionImpl();

        for (int i = 0, neuronsSize = outputHNeurons.size(); i < neuronsSize; ++i) {
            ActiveNeuron neuron = (ActiveNeuron) outputHNeurons.get(i);
            neuron.setDelta(calcDerivative(functionImpl, neuron) * (neuron.getActivatedOutput() - outputData[i]));
            correctWeights(neuron);
        }
    }

    private void recalculateHiddenLayerWeights() {
        for (int i = hiddenLayers.size() - 1; i >= 0; --i) {
            ActiveLayer hiddenLayer = (ActiveLayer) hiddenLayers.get(i);

            List<Neuron> neurons = hiddenLayer.getNeurons();
            Function functionImpl = hiddenLayer.getFunctionImpl();

            for (Neuron neuron : neurons) {
                ActiveNeuron layerNeuron = (ActiveNeuron) neuron;
                float derivative = calcDerivative(functionImpl, layerNeuron);

                ActiveLayer nextLayer = (ActiveLayer) ((i == hiddenLayers.size() - 1) ? outputLayer : hiddenLayers.get(i + 1));
                List<Neuron> nextLayerNeurons = nextLayer.getNeurons();

                float correction = 0;
                for (Neuron nextLayerNeuron : nextLayerNeurons) {
                    ActiveNeuron activeNeuron = (ActiveNeuron) nextLayerNeuron;

                    Map<Neuron, Float> connections = activeNeuron.getConnections();

                    if (connections.containsKey(layerNeuron)) {
                        correction += activeNeuron.getDelta() * connections.get(layerNeuron);
                    }
                }

                layerNeuron.setDelta(correction * derivative);
                if (i > 0) {
                    correctWeights(layerNeuron);
                } else {
                    correctLastWeights(layerNeuron);
                }
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
            calcOutput[i] = ((ActiveNeuron) neurons.get(i)).getActivatedOutput();
        }

        avgE += applyRateFunction(outputData, calcOutput);
        exampleCount += 1;
    }

    public void trainExample(List<Float> inputData, float[] outputData) {
        if (inputData.size() != inputLayer.getNeurons().size()) {
            throw new IllegalStateException("Incorrect data dimension for input layer");
        } else if (outputData.length != outputLayer.getNeurons().size()) {
            throw new IllegalStateException("Incorrect data dimension for output layer");
        } else {

            List<Neuron> neurons = inputLayer.getNeurons();

            for (int i = 0, neuronsSize = neurons.size(); i < neuronsSize; ++i) {
                ((InputNeuron) neurons.get(i)).setOutput(inputData.get(i));
            }

            calcFirstHiddenLayer();
            calcOtherHiddenLayers();
            calcOutputLayerOutputs();

            recalculateOutputLayerWeights(outputData);
            recalculateHiddenLayerWeights();

            calcError(outputData);
        }
    }

    public float getAverageError() {
        float E = avgE / exampleCount;

        exampleCount = 0;
        avgE = 0;

        return E;
    }

    public void printOutputs() {
        List<Neuron> neurons = outputLayer.getNeurons();
        System.out.print("neurons output: ");
        for (Neuron neuron : neurons) {
            System.out.print(((ActiveNeuron) neuron).getActivatedOutput() + " ");
        }
    }
}

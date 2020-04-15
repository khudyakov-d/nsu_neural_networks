package khudyakov_udod.perceptron.network;

import khudyakov_udod.perceptron.network.entities.layers.ActiveLayer;
import khudyakov_udod.perceptron.network.entities.layers.Layer;
import khudyakov_udod.perceptron.network.entities.neurons.ActiveNeuron;
import khudyakov_udod.perceptron.network.entities.neurons.InputNeuron;
import khudyakov_udod.perceptron.network.entities.neurons.Neuron;
import khudyakov_udod.perceptron.functions.Function;

import java.util.*;

public class NetBuilder {
    private static final Random rand = new Random();
    private static int idGenerator = 0;

    private final int inputNeuronsCounts;
    private final int outputNeuronsCounts;
    private final int hiddenLayersCount;

    public NetBuilder(int inputNeuronsCounts, int outputNeuronsCounts, int hiddenLayersCount) {
        this.inputNeuronsCounts = inputNeuronsCounts;
        this.outputNeuronsCounts = outputNeuronsCounts;
        this.hiddenLayersCount = hiddenLayersCount;
    }

    private Layer createInputLayer() {
        List<Neuron> inputHNeurons = new ArrayList<>();
        for (int i = 0; i < inputNeuronsCounts; i++) {
            inputHNeurons.add(new InputNeuron(idGenerator++));
        }
        return new Layer(inputHNeurons);
    }

    private Neuron createNeuronWithConnections(Layer layer) {
        Map<Neuron, Float> connectionWeights = new HashMap<>();
        for (Neuron neuron : layer.getNeurons()) {
            connectionWeights.put(neuron, rand.nextFloat() - 0.5f);
        }
        return new ActiveNeuron(idGenerator++, connectionWeights);
    }

    private List<Layer> createHiddenLayers(Layer inputLayer, List<Integer> hiddenLayerSizes, List<Function> functions) {

        List<Layer> hiddenLayers = new ArrayList<>();

        if (hiddenLayerSizes.size() != hiddenLayersCount || functions.size() != hiddenLayersCount) {
            throw new IllegalStateException("incorrect initialization of layer sizes");
        } else {
            for (int i = 0; i < hiddenLayersCount; i++) {
                Layer layer = (i == 0) ? inputLayer : hiddenLayers.get(i - 1);
                List<Neuron> neurons = new ArrayList<>();

                for (int j = 0; j < hiddenLayerSizes.get(i); j++) {
                    neurons.add(createNeuronWithConnections(layer));
                }

                hiddenLayers.add(new ActiveLayer(neurons, functions.get(i), null));
            }
        }

        return hiddenLayers;
    }

    private ActiveLayer createOutputLayer(Layer lastHiddenActiveLayer, Function outputFunction) {
        List<Neuron> outputHNeurons = new ArrayList<>();
        for (int i = 0; i < outputNeuronsCounts; i++) {
            outputHNeurons.add(createNeuronWithConnections(lastHiddenActiveLayer));
        }
        return new ActiveLayer(outputHNeurons, outputFunction, null);
    }

    public Net createNet(List<Integer> hiddenLayerSizes, List<Function> functions, float rate) {
        if (functions.size() != hiddenLayersCount + 1) {
            throw new IllegalStateException("Not correct count of activation functions");
        } else {

            Layer inputLayer = createInputLayer();
            List<Layer> hiddenLayers = createHiddenLayers(inputLayer, hiddenLayerSizes, functions.subList(0, hiddenLayersCount));
            Layer outputLayer = createOutputLayer(hiddenLayers.get(hiddenLayers.size() - 1), functions.get(functions.size() - 1));

            return new Net(inputLayer, hiddenLayers, outputLayer, rate);
        }
    }
}

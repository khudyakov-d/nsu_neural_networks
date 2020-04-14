package khudyakov_udod.perceptron.network;

import khudyakov_udod.perceptron.network.entities.ActiveLayer;
import khudyakov_udod.perceptron.network.entities.Layer;
import khudyakov_udod.perceptron.network.entities.Neuron;
import khudyakov_udod.perceptron.new_functions.Function;

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
        List<Neuron> inputNeurons = new ArrayList<>();
        for (int i = 0; i < inputNeuronsCounts; i++) {
            inputNeurons.add(new Neuron(idGenerator++));
        }
        return new Layer(inputNeurons);
    }

    private Neuron createNeuronWithConnections(Layer layer) {
        Map<Neuron, Float> connectionWeights = new HashMap<>();
        for (Neuron neuron : layer.getNeurons()) {
            connectionWeights.put(neuron, 0f);
        }
        return new Neuron(idGenerator++, connectionWeights);
    }

    private List<ActiveLayer> createHiddenLayers(Layer inputLayer, List<Integer> hiddenLayerSizes, List<Function> functions) {

        List<ActiveLayer> hiddenLayers = new ArrayList<>();

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

    private ActiveLayer createOutputLayer(ActiveLayer lastHiddenActiveLayer, Function outputFunction) {
        List<Neuron> outputNeurons = new ArrayList<>();
        for (int i = 0; i < outputNeuronsCounts; i++) {
            outputNeurons.add(createNeuronWithConnections(lastHiddenActiveLayer));
        }
        return new ActiveLayer(outputNeurons, outputFunction, null);
    }

    public Net createNet(List<Integer> hiddenLayerSizes, List<Function> functions, float rate) {
        if (functions.size() != hiddenLayersCount + 1) {
            throw new IllegalStateException("Not correct count of activation functions");
        } else {

            Layer inputLayer = createInputLayer();
            List<ActiveLayer> hiddenLayers = createHiddenLayers(inputLayer, hiddenLayerSizes, functions.subList(0, hiddenLayersCount));
            ActiveLayer outputLayer = createOutputLayer(hiddenLayers.get(hiddenLayers.size() - 1), functions.get(functions.size() - 1));

            return new Net(inputLayer, hiddenLayers, outputLayer, rate);
        }
    }
}

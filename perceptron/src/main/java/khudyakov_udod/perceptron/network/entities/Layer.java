package khudyakov_udod.perceptron.network.entities;

import java.util.List;

public class Layer {
    protected final List<Neuron> neurons;

    public Layer(List<Neuron> neurons) {
        this.neurons = neurons;
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }
}

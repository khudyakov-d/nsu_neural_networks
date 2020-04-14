package khudyakov_udod.perceptron.network.entities.layers;

import khudyakov_udod.perceptron.network.entities.neurons.Neuron;
import khudyakov_udod.perceptron.functions.Function;

import java.util.ArrayList;
import java.util.List;

public class ActiveLayer extends Layer {
    private final Function functionImpl;
    private final List<Float> bias;

    public ActiveLayer(List<Neuron> neurons, Function functionImpl, List<Float> bias) {
        super(neurons);
        this.functionImpl = functionImpl;

        if (null == bias) {
            this.bias = new ArrayList<>();
            for (int i = 0; i < neurons.size(); i++) {
                this.bias.add((float) 0);
            }
        } else {
            this.bias = bias;
        }
    }

    public Function getFunctionImpl() {
        return functionImpl;
    }

    public List<Float> getBias() {
        return bias;
    }
}

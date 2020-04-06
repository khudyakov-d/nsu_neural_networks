package khudyakov_udod.perceptron.network.entities;

import khudyakov_udod.perceptron.functions.FunctionImpl;

import java.util.ArrayList;
import java.util.List;

public class ActiveLayer extends Layer {
    private final FunctionImpl functionImpl;
    private final List<Float> bias;

    public ActiveLayer(List<Neuron> neurons, FunctionImpl functionImpl, List<Float> bias) {
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

    public FunctionImpl getFunctionImpl() {
        return functionImpl;
    }

    public List<Float> getBias() {
        return bias;
    }
}

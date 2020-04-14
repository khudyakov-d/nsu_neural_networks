package khudyakov_udod.perceptron.network.entities.neurons;

public class InputNeuron extends Neuron {
    private Float output;

    public InputNeuron(int id) {
        super(id);
    }

    public Float getOutput() {
        return output;
    }

    public void setOutput(Float output) {
        this.output = output;
    }
}

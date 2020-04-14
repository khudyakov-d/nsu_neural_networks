package khudyakov_udod.perceptron.network.entities.neurons;

import java.util.HashMap;
import java.util.Map;

public class ActiveNeuron extends Neuron {
    private float pureOutput;
    private float activatedOutput;
    private float delta;

    private Map<Neuron, Float> connections;

    public ActiveNeuron(int id, Map<Neuron, Float> connections) {
        super(id);
        this.connections = connections;
    }

    public float getActivatedOutput() {
        return activatedOutput;
    }

    public void setActivatedOutput(float activatedOutput) {
        this.activatedOutput = activatedOutput;
    }

    public float getPureOutput() {
        return pureOutput;
    }

    public void setPureOutput(float pureOutput) {
        this.pureOutput = pureOutput;
    }

    public float getDelta() {
        return delta;
    }

    public void setDelta(float delta) {
        this.delta = delta;
    }

    public Map<Neuron, Float> getConnections() {
        return connections;
    }

    public void setConnections(Map<Neuron, Float> connections) {
        this.connections = connections;
    }
}

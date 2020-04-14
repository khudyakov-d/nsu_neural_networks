package khudyakov_udod.perceptron.network.entities;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class Neuron {
    private final int id;

    private float pureOutput;
    private float activatedOutput;
    private float delta;

    private Map<Neuron, Float> connections = new HashMap<>();

    public Neuron(int id) {
        this.id = id;
    }

    public Neuron(int id, Map<Neuron, Float> connections) {
        this.id = id;
        this.connections = connections;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Neuron neuron = (Neuron) o;
        return id == neuron.id;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
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


}

package khudyakov_udod.perceptron.network.entities.neurons;

import java.util.Objects;

abstract public class Neuron {
    private final int id;

    public Neuron(int id) {
        this.id = id;
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
}

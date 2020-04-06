package khudyakov_udod.perceptron.functions;

public interface ActivationFunction {
        float applyFunc(float x, float a);
        float applySimplifiedDeriveFunc(float f, float a);
        float applyDeriveFunc(float f, float a);
}

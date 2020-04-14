package khudyakov_udod.perceptron.new_functions;

public class ReLuFunction extends Function {

    public ReLuFunction() {
        super(false);
    }

    @Override
    public float applyFunc(float x) {
        return x > 0 ? x : 0;
    }

    @Override
    public float applyDerivativeFunc(float x) {
        return x > 0 ? 1 : 0;
    }

    @Override
    public float applySimplifiedDerivativeFunc(float f) {
        return 0;
    }
}

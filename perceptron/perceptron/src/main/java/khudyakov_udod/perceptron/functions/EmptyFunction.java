package khudyakov_udod.perceptron.functions;

public class EmptyFunction extends Function {

    public EmptyFunction() {
        super(false);
    }

    @Override
    public float applyFunc(float x) {
        return x;
    }

    @Override
    public float applySimplifiedDerivativeFunc(float f) {
        return 0;
    }

    @Override
    public float applyDerivativeFunc(float x) {
        return x;
    }
}

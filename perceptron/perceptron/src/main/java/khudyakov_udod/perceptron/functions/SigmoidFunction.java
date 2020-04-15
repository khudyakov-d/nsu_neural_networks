package khudyakov_udod.perceptron.functions;

public class SigmoidFunction extends Function {
    private final float a;

    public SigmoidFunction(float a) {
        super(true);
        this.a = a;
    }

    @Override
    public float applyFunc(float x) {
        return 1 / (1 + (float) Math.exp(-x * a));
    }

    @Override
    public float applySimplifiedDerivativeFunc(float f) {
        return a * f * (1 - f);
    }

    @Override
    public float applyDerivativeFunc(float x) {
        float f = applyFunc(x);
        return a * f * (1 - f);
    }
}

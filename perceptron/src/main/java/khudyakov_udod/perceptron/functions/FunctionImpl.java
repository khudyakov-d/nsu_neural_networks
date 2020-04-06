package khudyakov_udod.perceptron.functions;

public class FunctionImpl {
    private final ActivationFunction function;
    private final float a;
    private final boolean simplifyAvailable;

    public FunctionImpl(ActivationFunction function, float a, boolean simplifyAvailable) {
        this.function = function;
        this.a = a;
        this.simplifyAvailable = simplifyAvailable;
    }

    public ActivationFunction getFunction() {
        return function;
    }

    public float getA() {
        return a;
    }

    public boolean isSimplifyAvailable() {
        return simplifyAvailable;
    }
}

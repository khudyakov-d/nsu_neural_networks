package khudyakov_udod.perceptron.functions;

public abstract class Function {
    private boolean simplifyAvailable;

    Function(boolean simplifyAvailable) {
        this.simplifyAvailable = simplifyAvailable;
    }

    public abstract float applyFunc(float x);

    public abstract float applySimplifiedDerivativeFunc(float f);

    public abstract float applyDerivativeFunc(float x);

    public boolean isSimplifyAvailable() {
        return simplifyAvailable;
    }
}

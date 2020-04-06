package khudyakov_udod.perceptron.functions;

public class FunctionsBuilder {

    public static FunctionImpl buildSigmoidFunction(float a) {
        return new FunctionImpl(buildSigmoid(), a, true);
    }

    private static ActivationFunction buildSigmoid() {
        return new ActivationFunction() {
            @Override
            public float applyFunc(float x, float a) {
                return 1 / (1 + (float) Math.exp(-x * a));
            }

            @Override
            public float applySimplifiedDeriveFunc(float f, float a) {
                return a * f * (1 - f);
            }

            @Override
            public float applyDeriveFunc(float x, float a) {
                float f = applyFunc(x, a);
                return a * f * (1 - f);
            }
        };
    }

    public static FunctionImpl buildReLuFunction() {
        return new FunctionImpl(buildSReLu(), 0, false);
    }

    private static ActivationFunction buildSReLu() {
        return new ActivationFunction() {
            @Override
            public float applyFunc(float x, float a) {
                return x > 0 ? x : 0;
            }

            @Override
            public float applySimplifiedDeriveFunc(float f, float a) {
                return 0;
            }

            @Override
            public float applyDeriveFunc(float x, float a) {
                return x > 0 ? 1 : 0;
            }
        };
    }

    public static FunctionImpl buildNoFunction() {
        return new FunctionImpl(buildSReLu(), 0, false);
    }

    private static ActivationFunction buildNo() {
        return new ActivationFunction() {
            @Override
            public float applyFunc(float x, float a) {
                return x;
            }

            @Override
            public float applySimplifiedDeriveFunc(float f, float a) {
                return 0;
            }

            @Override
            public float applyDeriveFunc(float x, float a) {
                return x;
            }
        };
    }
}


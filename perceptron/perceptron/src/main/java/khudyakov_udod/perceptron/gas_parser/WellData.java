package khudyakov_udod.perceptron.gas_parser;

import java.util.List;

public class WellData {
    private List<Float> data;
    private float gTotal;
    private float KGF;

    public WellData(List<Float> data, float gTotal, float KGF) {
        this.data = data;
        this.gTotal = gTotal;
        this.KGF = KGF;
    }

    public List<Float> getData() {
        return data;
    }

    public float[] convertOutputs() {
        float[] outputs = new float[2];
        outputs[0] = gTotal;
        outputs[1] = KGF;

        return outputs;
    }


    public int getDataSize() {
        return data.size();
    }

    public float getgTotal() {
        return gTotal;
    }

    public float getKGF() {
        return KGF;
    }
}

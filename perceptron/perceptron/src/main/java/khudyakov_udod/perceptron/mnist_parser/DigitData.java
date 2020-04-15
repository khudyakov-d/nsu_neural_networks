package khudyakov_udod.perceptron.mnist_parser;

import java.util.ArrayList;
import java.util.List;

public class DigitData {
    private final float[] data;
    private final byte label;
    private final int rowsCount;
    private final int columnsCount;

    public DigitData(byte label, int rowsCount, int columnsCount) {
        this.label = label;
        this.rowsCount = rowsCount;
        this.columnsCount = columnsCount;
        data = new float[rowsCount * columnsCount];
    }

    public float[] getData() {
        return data;
    }

    public List<Float> getDataAsList() {
        List<Float> list = new ArrayList<>(data.length);

        for (float el : data) {
            list.add(el);
        }

        return list;
    }

    public void setDataValue(int pos, float value) {
        if (pos > rowsCount * columnsCount - 1) {
            throw new IllegalStateException("Position isn't in diapason");
        } else {
            data[pos] = value;
        }
    }

    public float[] convertLabelToArray() {
        float[] labelData = new float[10];
        for (int i = 0; i < 10; i++) {
            if (i == label) {
                labelData[i] = 1;
            } else {
                labelData[i] = 0;
            }
        }
        return labelData;
    }
}
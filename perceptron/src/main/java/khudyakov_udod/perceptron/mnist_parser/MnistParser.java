package khudyakov_udod.perceptron.mnist_parser;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MnistParser {
    private final String imagesFilePath;
    private final String labelsFilePath;

    public MnistParser(String imagesFilePath, String labelsFilePath) {
        this.imagesFilePath = imagesFilePath;
        this.labelsFilePath = labelsFilePath;
    }

    public List<DigitData> readDigits() throws IOException {
        List<DigitData> digitDataList = new ArrayList<>();

        try (
                DataInputStream labelsInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(labelsFilePath)));
                DataInputStream imagesInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(imagesFilePath)))
        ) {
            int labelsMagicNumber = labelsInputStream.readInt();
            int imagesMagicNumber = imagesInputStream.readInt();

            int labelsCount = labelsInputStream.readInt();
            int imagesCount = imagesInputStream.readInt();

            if (labelsCount != imagesCount) {
                throw new IllegalStateException("The number of labels and images is different");
            } else {
                int rowsCount = imagesInputStream.readInt();
                int columnsCount = imagesInputStream.readInt();
                int imageSize = rowsCount * columnsCount;

                for (int i = 0; i < imagesCount; i++) {
                    DigitData digitData = new DigitData(labelsInputStream.readByte(), rowsCount, columnsCount);

                    for (int j = 0; j < imageSize; j++) {
                        byte b = imagesInputStream.readByte();
                        digitData.setDataValue(j, (b & 0xFF) / 255f);
                    }

                    digitDataList.add(digitData);
                }
            }
        }

        return digitDataList;
    }


}

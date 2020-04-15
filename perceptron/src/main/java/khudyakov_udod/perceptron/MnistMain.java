package khudyakov_udod.perceptron;

import khudyakov_udod.perceptron.functions.Function;
import khudyakov_udod.perceptron.functions.SigmoidFunction;
import khudyakov_udod.perceptron.mnist_parser.DigitData;
import khudyakov_udod.perceptron.mnist_parser.MnistParser;
import khudyakov_udod.perceptron.network.Net;
import khudyakov_udod.perceptron.network.NetBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

class MnistMain {

    @SuppressWarnings("Duplicates")
    public static void main(String[] args) throws IOException {
        Random rand = new Random();

        MnistParser mnistParser = new MnistParser("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
        List<DigitData> digitDataList = mnistParser.readDigits();

        Function sigmoidFunction = new SigmoidFunction(1);

        List<Function> functions = new ArrayList<>();
        functions.add(sigmoidFunction);
        functions.add(sigmoidFunction);
        functions.add(sigmoidFunction);

        NetBuilder netBuilder = new NetBuilder(28 * 28, 10, 2);

        Net net = netBuilder.createNet(Arrays.asList(64, 32), functions, 0.05f);

        List<DigitData> trainDigitData = digitDataList.subList(0, (digitDataList.size() / 10) * 7);
        List<DigitData> testDigitData = digitDataList.subList((digitDataList.size() / 10) * 7, digitDataList.size());

        int batchSize = 1000;
        int[] indexes;
        int epochCount = 500;

        for (int i = 0; i < epochCount; ++i) {
            indexes = rand.ints(batchSize, 0, trainDigitData.size()).toArray();

            for (int j = 0; j < batchSize; j++) {
                net.trainExample(trainDigitData.get(indexes[j]).getDataAsList(), trainDigitData.get(indexes[j]).convertLabelToArray());
            }

            float error = net.getAverageError();
            System.out.println("epoch number #" + i + " avg error: " + error);
            if (error <= 0.001) {
                break;
            }
        }

        for (DigitData testDigitDatum : testDigitData) {
            net.predictExample(testDigitDatum.getDataAsList(), testDigitDatum.convertLabelToArray());
        }

        float error = net.getAverageError();
        System.out.println();
        System.out.println("test avg error: " + error);
    }
}

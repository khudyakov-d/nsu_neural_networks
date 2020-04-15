package khudyakov_udod.perceptron;

import khudyakov_udod.perceptron.functions.Function;
import khudyakov_udod.perceptron.functions.SigmoidFunction;
import khudyakov_udod.perceptron.gas_parser.WellData;
import khudyakov_udod.perceptron.gas_parser.WellParser;
import khudyakov_udod.perceptron.network.Net;
import khudyakov_udod.perceptron.network.NetBuilder;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class WellMain {

    @SuppressWarnings("Duplicates")
    public static void main(String[] args) throws IOException {
        Random rand = new Random();

        WellParser wellParser = new WellParser("well_data.csv");
        List<WellData> wellDataList = wellParser.readData();

        Function sigmoidFunction = new SigmoidFunction(1);
        List<Function> functions = new ArrayList<>();
        functions.add(sigmoidFunction);
        functions.add(sigmoidFunction);
        functions.add(sigmoidFunction);

        NetBuilder netBuilder = new NetBuilder(wellDataList.get(0).getDataSize(), 2, 2);
        Net net = netBuilder.createNet(Arrays.asList(64, 32), functions, 0.01f);

        List<WellData> trainDigitData = wellDataList.subList(0, (wellDataList.size() / 10) * 7);
        List<WellData> testDigitData = wellDataList.subList((wellDataList.size() / 10) * 7, wellDataList.size());

        int batchSize = 20;
        int[] indexes;
        int i = 0;

        while (true) {
            indexes = rand.ints(batchSize, 0, trainDigitData.size()).toArray();

            for (int j = 0; j < batchSize; j++) {
                net.trainExample(trainDigitData.get(indexes[j]).getData(), trainDigitData.get(indexes[j]).convertOutputs());
            }

            float error = net.getAverageError();
            System.out.println("epoch number #" + i++ + " avg error: " + error);
            if (error <= 0.001) {
                break;
            }
        }

        for (WellData testDigitDatum : testDigitData) {
            net.predictExample(testDigitDatum.getData(), testDigitDatum.convertOutputs());
        }

        float error = net.getAverageError();
        System.out.println();
        System.out.println("test avg error: " + error);
    }
}

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

class Main {
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

        Net net = netBuilder.createNet(Arrays.asList(64, 32), functions, 0.1f);


        int batchSize = 1000;
        int[] indexes;
        int epochCount = 250;

        for (int i = 0; i < epochCount; ++i) {
            indexes = rand.ints(batchSize, 0, digitDataList.size() + 1).toArray();

            for (int j = 0; j < batchSize; j++) {
                net.trainExample((digitDataList.get(indexes[i]).getDataAsList()), digitDataList.get(indexes[i]).convertLabelToArray());

                if (j == batchSize - 1) {

                    float[] answer = digitDataList.get(indexes[i]).convertLabelToArray();
                    System.out.print("right output: ");
                    for (int k = 0; k < answer.length; k++) {
                        System.out.print(answer[k] + " ");
                    }
                    System.out.println();
                    net.printOutputs();
                }
            }

            float error = net.getAverageError();
            System.out.println();
            System.out.println("avg error: " + error);
        }
    }
}

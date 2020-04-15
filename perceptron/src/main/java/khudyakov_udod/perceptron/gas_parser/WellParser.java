package khudyakov_udod.perceptron.gas_parser;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class WellParser {
    private final String filePath;
    private List<String> headerNames;
    private Map<Integer, Float> maxValues;

    public WellParser(String filePath) {
        this.filePath = filePath;
    }

    private List<WellData> convertToWellData(List<List<Float>> dataLists) {
        List<WellData> wellDataList = new ArrayList<>();

        for (List<Float> dataList : dataLists) {
            WellData wellData = new WellData(
                    dataList.subList(0, dataList.size() - 2),
                    dataList.get(dataList.size() - 2),
                    dataList.get(dataList.size() - 1));

            wellDataList.add(wellData);
        }

        return wellDataList;
    }

    private List<List<Float>> fillData(List<CSVRecord> csvRecords) {
        List<List<Float>> dataLists = new ArrayList<>(csvRecords.size());

        for (CSVRecord csvRecord : csvRecords) {
            List<Float> dataList = new ArrayList<>();

            for (int j = 0, size = csvRecord.size(); j < size; j++) {
                String stringValue = csvRecord.get(j);
                if (stringValue.equals("")) {
                    dataList.add(null);
                } else {
                    float curValue = Float.valueOf(csvRecord.get(j).replace(",","."));
                    if (curValue > maxValues.get(j)) {
                        maxValues.put(j, curValue);
                    }
                    dataList.add(curValue);
                }
            }
            dataLists.add(dataList);
        }

        return dataLists;
    }

    private void normalizeData(List<List<Float>> dataLists) {
        for (List<Float> dataList : dataLists) {
            for (int i = 0, size = dataList.size(); i < size; ++i) {
                if (dataList.get(i) != null) {
                    dataList.set(i, dataList.get(i) / maxValues.get(i));
                }
            }
        }
    }

    public List<WellData> readData() {
        try (
                FileReader fileReader = new FileReader(filePath);
                CSVParser parser = CSVParser.parse(fileReader, CSVFormat.EXCEL.withDelimiter(',').withHeader())
        ) {
            headerNames = parser.getHeaderNames();

            List<CSVRecord> records = new ArrayList<>();
            for (CSVRecord csvRecord : parser) {
                records.add(csvRecord);
            }

            maxValues = new HashMap<>();
            int size = parser.getHeaderNames().size();
            for (int i = 0; i < size; i++) {
                maxValues.put(i, 0f);
            }

            List<List<Float>> dataLists = fillData(records);
            normalizeData(dataLists);

            return convertToWellData(dataLists);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }

    public List<String> getHeaderNames() {
        return headerNames;
    }

    public Map<Integer, Float> getMaxValues() {
        return maxValues;
    }
}


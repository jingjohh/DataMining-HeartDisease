
package heartdisease.Preprocessing;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

public class DataPreprocessing {

    private static final String TARGET_ATTRIBUTE = "Heart Disease Status";

    // Store imputation values learned from training data
    public static class Imputer {

        private double[] replacementValues;
        private boolean[] hasReplacementValue;

        private Imputer(int numAttributes) {
            replacementValues = new double[numAttributes];
            hasReplacementValue = new boolean[numAttributes];
        }
    }

    // Remove records with missing target values
    public static Instances removeMissingTarget(Instances dataset) {

        System.out.println("\n--- REMOVE MISSING TARGET ---");

        Instances processedDataset = new Instances(dataset);

        Attribute target = processedDataset.attribute(TARGET_ATTRIBUTE);

        if (target == null) {
            throw new IllegalArgumentException(
                    "Target attribute not found: " + TARGET_ATTRIBUTE
            );
        }

        if (!target.isNominal()) {
            throw new IllegalArgumentException(
                    "Target attribute must be categorical."
            );
        }

        int targetIndex = target.index();
        int removedCount = 0;

        // Traverse backwards to safely delete records
        for (int i = processedDataset.numInstances() - 1; i >= 0; i--) {

            if (processedDataset.instance(i).isMissing(targetIndex)) {

                processedDataset.delete(i);
                removedCount++;
            }
        }

        processedDataset.setClassIndex(targetIndex);

        System.out.println("Original Records: " + dataset.numInstances());
        System.out.println("Removed Records: " + removedCount);
        System.out.println("Remaining Records: " + processedDataset.numInstances());

        return processedDataset;
    }

    // Calculate Median of numerical attributes
    private static double calculateMedian(ArrayList<Double> values) {

        Collections.sort(values);

        int size = values.size();

        if (size % 2 == 1) {
            return values.get(size / 2);
        }

        return (values.get(size / 2 - 1) + values.get(size / 2)) / 2.0;
    }

    // Calculate Mode of categorical attributes
    private static double calculateMode(AttributeStats stats) {

        int[] counts = stats.nominalCounts;

        int modeIndex = -1;
        int maxCount = 0;

        for (int i = 0; i < counts.length; i++) {

            if (counts[i] > maxCount) {
                maxCount = counts[i];
                modeIndex = i;
            }
        }

        if (modeIndex == -1) {
            throw new IllegalStateException(
                    "Cannot calculate mode: attribute has no valid values."
            );
        }

        return modeIndex;
    }

    // Learn Median and Mode from training data only
    public static Imputer fitImputer(Instances trainingData) {

        System.out.println("\n--- FIT MISSING VALUE IMPUTER ---");

        Imputer imputer = new Imputer(trainingData.numAttributes());

        int targetIndex = trainingData.attribute(TARGET_ATTRIBUTE).index();

        for (int i = 0; i < trainingData.numAttributes(); i++) {

            Attribute attribute = trainingData.attribute(i);

            // Do not impute target attribute
            if (i == targetIndex) {
                continue;
            }

            // Handle numerical attributes using Median
            if (attribute.isNumeric()) {

                ArrayList<Double> values = new ArrayList<>();

                for (int j = 0; j < trainingData.numInstances(); j++) {

                    Instance instance = trainingData.instance(j);

                    if (!instance.isMissing(i)) {
                        values.add(instance.value(i));
                    }
                }

                if (!values.isEmpty()) {

                    double median = calculateMedian(values);

                    imputer.replacementValues[i] = median;
                    imputer.hasReplacementValue[i] = true;

                    System.out.println(
                            attribute.name() + " | Median: " + median
                    );
                }
            }

            // Handle categorical attributes using Mode
            else if (attribute.isNominal()) {

                AttributeStats stats = trainingData.attributeStats(i);

                double modeIndex = calculateMode(stats);

                imputer.replacementValues[i] = modeIndex;
                imputer.hasReplacementValue[i] = true;

                System.out.println(
                        attribute.name() + " | Mode: "
                                + attribute.value((int) modeIndex)
                );
            }
        }

        return imputer;
    }

    // Apply learned Median and Mode to a dataset
    public static Instances transformMissingValues(
            Instances dataset,
            Imputer imputer
    ) {

        System.out.println("\n--- TRANSFORM MISSING VALUES ---");

        if (dataset.numAttributes() != imputer.replacementValues.length) {
            throw new IllegalArgumentException(
                    "Dataset structure does not match the fitted imputer."
            );
        }

        Instances processedDataset = new Instances(dataset);

        int targetIndex = processedDataset.attribute(TARGET_ATTRIBUTE).index();

        int totalReplaced = 0;

        for (int i = 0; i < processedDataset.numAttributes(); i++) {

            // Do not modify target attribute
            if (i == targetIndex) {
                continue;
            }

            if (!imputer.hasReplacementValue[i]) {
                continue;
            }

            for (int j = 0; j < processedDataset.numInstances(); j++) {

                Instance instance = processedDataset.instance(j);

                if (instance.isMissing(i)) {

                    instance.setValue(
                            i,
                            imputer.replacementValues[i]
                    );

                    totalReplaced++;
                }
            }
        }

        System.out.println("Missing Values Replaced: " + totalReplaced);

        return processedDataset;
    }
}

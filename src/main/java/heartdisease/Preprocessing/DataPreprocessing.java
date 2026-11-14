
package heartdisease.Preprocessing;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;

import weka.core.converters.ArffSaver;
import java.io.File;
import java.io.IOException;

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

    // Handling outlier
    // Calculate quartiles
    private static double[] calculateQuartiles(ArrayList<Double> values) {

        Collections.sort(values);

        int size = values.size();

        if (size < 2) {
            throw new IllegalArgumentException(
                    "At least two values are required."
            );
        }

        int middle = size / 2;

        ArrayList<Double> lowerHalf = new ArrayList<>(values.subList(0, middle));

        int upperStart;

        if (size % 2 == 0) {
            upperStart = middle;
        } else {
            upperStart = middle + 1;
        }

        ArrayList<Double> upperHalf = new ArrayList<>(values.subList(upperStart, size));

        // Calculate Q1 and Q3 using the median of two halves
        double q1 = calculateMedian(lowerHalf);
        double q3 = calculateMedian(upperHalf);

        return new double[]{q1, q3};
    }

    // OutlierBounds class
    // Store IQR boundaries for each numerical attribute
    public static class OutlierBounds {

        private double[] lowerBounds;
        private double[] upperBounds;
        private boolean[] hasBounds;

        public OutlierBounds(int numAttributes) {
            // Each index corresponds to an attribute in the dataset
            lowerBounds = new double[numAttributes];
            upperBounds = new double[numAttributes];
            hasBounds = new boolean[numAttributes];
        }
    }

    // Fit outlier boundaries using training data
    public static OutlierBounds fitOutlierBounds(Instances trainingData) {

        System.out.println("\n--- FIT OUTLIER BOUNDS ---");
        
        // Create storage for the lower and upper bounds
        OutlierBounds bounds = new OutlierBounds(trainingData.numAttributes());

        Attribute target = trainingData.attribute(TARGET_ATTRIBUTE);

        if (target == null) {
            throw new IllegalArgumentException(
                "Target attribute not found."
            );
        }

        int targetIndex = target.index();

        for (int i = 0; i < trainingData.numAttributes(); i++) {

            Attribute attribute = trainingData.attribute(i);

            // Skip the target and non-numerical attributes
            if (i == targetIndex || !attribute.isNumeric()) {
                continue;
            }

            ArrayList<Double> values = new ArrayList<>();

            for (int j = 0;
                j < trainingData.numInstances(); j++) {

                Instance instance = trainingData.instance(j);
                
                // Collect only non-missing values
                if (!instance.isMissing(i)) {
                        values.add(instance.value(i));
                }
            }

            // Quartiles require at least two values
            if (values.size() < 2) {
                System.out.println( attribute.name() + ": Insufficient data");
                continue;
            }

            double[] quartiles = calculateQuartiles(values);

            double q1 = quartiles[0];
            double q3 = quartiles[1];
            
            double iqr = q3 - q1;       // Calculate Interquartile Range

            // Calculate the lower and upper outlier boundaries
            double lower = q1 - 1.5 * iqr;
            double upper = q3 + 1.5 * iqr;

            bounds.lowerBounds[i] = lower;
            bounds.upperBounds[i] = upper;
            bounds.hasBounds[i] = true;

            System.out.printf("%s | Q1: %.2f | Q3: %.2f | IQR: %.2f" + " | Lower: %.2f | Upper: %.2f%n", attribute.name(), q1, q3, iqr, lower, upper);
        }

        return bounds;
    }

    // Outlier Analysis
    public static void analyzeOutliers(Instances dataset, OutlierBounds bounds) {

        System.out.println("\n--- OUTLIER ANALYSIS ---");

        // Ensure the dataset has the same number of attributes
        if (dataset.numAttributes() != bounds.lowerBounds.length) {
            throw new IllegalArgumentException(
                    "Dataset structure does not match outlier bounds."
            );
        }

        Attribute target = dataset.attribute(TARGET_ATTRIBUTE);

        if (target == null) {
            throw new IllegalArgumentException(
                    "Target attribute not found."
            );
        }

        int targetIndex = target.index();

        for (int i = 0; i < dataset.numAttributes(); i++) {

            Attribute attribute = dataset.attribute(i);

             // Skip the target and non-numerical attributes
            if (i == targetIndex || !attribute.isNumeric()) {
                continue;
            }

            // Skip attributes without calculated boundaries
            if (!bounds.hasBounds[i]) {
                System.out.println(
                        attribute.name() + ": Cannot analyze"
                );
                continue;
            }

            int outlierCount = 0;
            int validCount = 0;

            for (int j = 0; j < dataset.numInstances(); j++) {

                Instance instance = dataset.instance(j);

                // Ignore missing values during analysis
                if (instance.isMissing(i)) {
                    continue;
                }

                validCount++;

                double value = instance.value(i);

                // A value outside either boundary is an outlier
                if (value < bounds.lowerBounds[i] || value > bounds.upperBounds[i]) {
                    outlierCount++;
                }
            }

            // Calculate the percentage of outliers
            double percentage;

            if (validCount == 0) {
                percentage = 0;
            } else {
                percentage = (double) outlierCount / validCount * 100;
            }

            System.out.printf(
                    "%-25s | Outliers: %d | Valid: %d | Rate: %.2f%%%n",
                    attribute.name(),
                    outlierCount,
                    validCount,
                    percentage
            );
        }
    }

    // Store mean and standard deviation for each numerical attribute
    public static class Standardizer {

        private double[] means;
        private double[] standardDeviations;
        private boolean[] hasStatistics;

        public Standardizer(int numAttributes) {
            means = new double[numAttributes];
            standardDeviations = new double[numAttributes];
            hasStatistics = new boolean[numAttributes];
        }
    }

    // Fit standardization statistics using training data
    public static Standardizer fitStandardizer(Instances trainingData) {

        System.out.println("\n--- FIT STANDARDIZER ---");

        Standardizer standardizer =
                new Standardizer(trainingData.numAttributes());

        Attribute target = trainingData.attribute(TARGET_ATTRIBUTE);

        if (target == null) {
            throw new IllegalArgumentException(
                    "Target attribute not found."
            );
        }

        int targetIndex = target.index();

        for (int i = 0; i < trainingData.numAttributes(); i++) {

            Attribute attribute = trainingData.attribute(i);

            // Skip target and non-numerical attributes
            if (i == targetIndex || !attribute.isNumeric()) {
                continue;
            }

            double sum = 0;
            int count = 0;

            // Calculate the mean
            for (int j = 0; j < trainingData.numInstances(); j++) {

                Instance instance = trainingData.instance(j);

                if (!instance.isMissing(i)) {
                    sum += instance.value(i);
                    count++;
                }
            }

            if (count < 2) {
                System.out.println(attribute.name() + ": Insufficient data");
                continue;
            }

            double mean = sum / count;

            double squaredDifferenceSum = 0;

            // Calculate the sum of squared differences
            for (int j = 0; j < trainingData.numInstances(); j++) {

                Instance instance = trainingData.instance(j);

                if (!instance.isMissing(i)) {

                    double difference = instance.value(i) - mean;

                    squaredDifferenceSum += difference * difference;
                }
            }

            // Calculate sample standard deviation
            double standardDeviation =
                    Math.sqrt(squaredDifferenceSum / (count - 1));

            // Avoid division by zero for constant attributes
            if (standardDeviation == 0) {
                System.out.println(attribute.name() + ": Standard deviation is zero");
                continue;
            }

            // Save statistics for later transformation
            standardizer.means[i] = mean;
            standardizer.standardDeviations[i] = standardDeviation;
            standardizer.hasStatistics[i] = true;

            System.out.printf(
                    "%s | Mean: %.2f | Std: %.2f%n",
                    attribute.name(), mean, standardDeviation
            );
        }
        return standardizer;
    }

    // Apply standardization using previously calculated statistics
    public static Instances transformStandardization(
            Instances dataset,
            Standardizer standardizer) {

        System.out.println("\n--- APPLY STANDARDIZATION ---");

        if (dataset.numAttributes() != standardizer.means.length) {
            throw new IllegalArgumentException(
                    "Dataset structure does not match standardizer."
            );
        }

        // Create a copy to preserve the original dataset
        Instances standardizedDataset = new Instances(dataset);

        Attribute target = standardizedDataset.attribute(TARGET_ATTRIBUTE);

        if (target == null) {
            throw new IllegalArgumentException(
                    "Target attribute not found."
            );
        }

        int targetIndex = target.index();

        int standardizedCount = 0;

        for (int i = 0; i < standardizedDataset.numAttributes(); i++) {

            Attribute attribute = standardizedDataset.attribute(i);

            // Skip target and non-numerical attributes
            if (i == targetIndex || !attribute.isNumeric()) {
                continue;
            }

            if (!standardizer.hasStatistics[i]) {
                continue;
            }

            double mean = standardizer.means[i];
            double standardDeviation =
                    standardizer.standardDeviations[i];

            for (int j = 0; j < standardizedDataset.numInstances(); j++) {

                Instance instance = standardizedDataset.instance(j);

                // Keep missing values unchanged
                if (instance.isMissing(i)) {
                    continue;
                }

                // Apply Z-score standardization
                double value = instance.value(i);

                double standardizedValue =
                        (value - mean) / standardDeviation;

                instance.setValue(i, standardizedValue);

                standardizedCount++;
            }

            System.out.println(attribute.name() + ": Standardized");
        }

        System.out.println(
                "Total standardized values: " + standardizedCount
        );

        return standardizedDataset;
    }

    // Convert dataset to ARFF format
    public static void saveAsArff(Instances dataset, String outputPath) throws IOException {
        
        System.out.println("\n--- ARFF CONVERSION ---");

        if (dataset == null || dataset.numInstances() == 0) {
            throw new IllegalArgumentException(
                    "Dataset is empty or null."
            );
        }

        // Check whether the target attribute exists
        Attribute target = dataset.attribute(TARGET_ATTRIBUTE);

        if (target == null) {
            throw new IllegalArgumentException(
                    "Target attribute not found."
            );
        }

        // Create output directory if it does not exist
        File outputFile = new File(outputPath);

        File parentDirectory = outputFile.getParentFile();

        if (parentDirectory != null && !parentDirectory.exists()) {
            parentDirectory.mkdirs();
        }

        // Create ARFF saver
        ArffSaver saver = new ArffSaver();

        saver.setInstances(dataset);
        saver.setFile(outputFile);

        // Write dataset to ARFF file
        saver.writeBatch();

        System.out.println("ARFF file saved successfully!");
        System.out.println("Output path: " + outputFile.getAbsolutePath());
        System.out.println("Number of instances: " + dataset.numInstances());
        System.out.println("Number of attributes: " + dataset.numAttributes());

    }
}

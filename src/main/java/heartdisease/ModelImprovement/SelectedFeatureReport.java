package heartdisease.ModelImprovement;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SelectedFeatureReport {

    private static final String DATA_PATH =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    private static final String OUTPUT_PATH =
            "results/step3_selected_features_training.csv";

    private static final int RANDOM_SEED = 42;

    public static void main(String[] args) throws Exception {

        // ============================================
        // STEP 1: LOAD DATASET
        // ============================================

        Instances data = DataSource.read(DATA_PATH);

        if (data == null) {
            throw new IllegalStateException(
                    "Cannot load dataset: " + DATA_PATH
            );
        }

        if (data.attribute(TARGET) == null) {
            throw new IllegalStateException(
                    "Target attribute not found: " + TARGET
            );
        }

        data.setClassIndex(
                data.attribute(TARGET).index()
        );

        // ============================================
        // STEP 2: REPRODUCE TRAINING SPLIT
        // ============================================

        data.randomize(new Random(RANDOM_SEED));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);

        // The held-out test set is not used.
        // Feature Selection uses training data only.

        System.out.println("================================");
        System.out.println("STEP 3 - SELECTED FEATURES REPORT");
        System.out.println("================================");

        System.out.println(
                "Training instances: "
                        + trainData.numInstances()
        );

        System.out.println(
                "Original features: "
                        + (trainData.numAttributes() - 1)
        );

        // ============================================
        // STEP 3: PREPARE CSV
        // ============================================

        StringBuilder csv = new StringBuilder();

        csv.append(
                "Method,SelectedFeatureCount,"
                        + "OriginalFeatureIndex,FeatureName,"
                        + "SelectionScope"
        ).append(System.lineSeparator());

        // ============================================
        // STEP 4: CFS + BESTFIRST
        // ============================================

        runSelection(
                "CFS + BestFirst",
                FeatureSelectionConfig.createEvaluator(),
                FeatureSelectionConfig.createSearchMethod(),
                trainData,
                csv
        );

        // ============================================
        // STEP 5: INFORMATION GAIN
        // ============================================

        for (int k :
                FeatureSelectionConfig.getTopKValues()) {

            if (k > trainData.numAttributes() - 1) {
                continue;
            }

            runSelection(
                    "InfoGain Top " + k,
                    FeatureSelectionConfig.createInfoGainEvaluator(),
                    FeatureSelectionConfig.createRanker(k),
                    trainData,
                    csv
            );
        }

        // ============================================
        // STEP 6: SAVE CSV
        // ============================================

        Path outputFile = Path.of(OUTPUT_PATH);

        Files.createDirectories(
                outputFile.getParent()
        );

        Files.writeString(
                outputFile,
                csv.toString(),
                StandardCharsets.UTF_8
        );

        System.out.println();
        System.out.println("================================");
        System.out.println("SELECTED FEATURES REPORT EXPORTED");
        System.out.println("================================");

        System.out.println(
                "File: " + outputFile.toAbsolutePath()
        );
    }

    private static void runSelection(
            String methodName,
            ASEvaluation evaluator,
            ASSearch search,
            Instances trainData,
            StringBuilder csv
    ) throws Exception {

        AttributeSelection selector =
                new AttributeSelection();

        selector.setEvaluator(evaluator);
        selector.setSearch(search);

        // Fit selection on all 8,000 training instances.
        // This report describes the resulting feature set.
        selector.SelectAttributes(trainData);

        int[] selectedIndices =
                selector.selectedAttributes();

        // Count features, excluding the target class.
        int selectedCount = 0;

        for (int index : selectedIndices) {
            if (index >= 0
                    && index < trainData.numAttributes()
                    && index != trainData.classIndex()) {
                selectedCount++;
            }
        }

        System.out.println();
        System.out.println("--- " + methodName + " ---");

        System.out.println(
                "Selected feature count: " + selectedCount
        );

        // ============================================
        // PRINT AND STORE FEATURE NAMES
        // ============================================

        for (int index : selectedIndices) {

            if (index < 0
                    || index >= trainData.numAttributes()
                    || index == trainData.classIndex()) {
                continue;
            }

            String featureName =
                    trainData.attribute(index).name();

            System.out.println(
                    "- " + featureName
            );

            csv.append(escapeCsv(methodName))
                    .append(",")
                    .append(selectedCount)
                    .append(",")
                    .append(index + 1)
                    .append(",")
                    .append(escapeCsv(featureName))
                    .append(",")
                    .append(escapeCsv("FullTraining"))
                    .append(System.lineSeparator());
        }
    }

    private static String escapeCsv(String value) {

        if (value == null) {
            return "";
        }

        return "\""
                + value.replace("\"", "\"\"")
                + "\"";
    }
}
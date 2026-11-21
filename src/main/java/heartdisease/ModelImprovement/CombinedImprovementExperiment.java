package heartdisease.ModelImprovement;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class CombinedImprovementExperiment {

    private static final String DATA_PATH =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    private static final String OUTPUT_PATH =
            "results/step3_combined_improvement_cv.csv";

    private static final int RANDOM_SEED = 42;
    private static final int CV_FOLDS = 10;

    private static final int[] TOP_K = {10, 15};
    private static final int[] YES_WEIGHTS = {4, 8};

    public static void main(String[] args) throws Exception {

        List<ImprovementResult> results = new ArrayList<>();

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

        if (!data.classAttribute().isNominal()
                || data.classAttribute().numValues() != 2) {
            throw new IllegalStateException(
                    "Expected a binary nominal target."
            );
        }

        int noIndex =
                data.classAttribute().indexOfValue("No");

        int yesIndex =
                data.classAttribute().indexOfValue("Yes");

        if (noIndex < 0 || yesIndex < 0) {
            throw new IllegalStateException(
                    "Target must contain No and Yes."
            );
        }

        // Reproduce Step 2 train/test split.
        data.randomize(new Random(RANDOM_SEED));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);

        // The held-out test set is not used
        // during Step 3 configuration selection.

        System.out.println("================================");
        System.out.println("STEP 3 - COMBINED MODEL IMPROVEMENT");
        System.out.println("================================");

        System.out.println(
                "Training instances: " + trainData.numInstances()
        );

        System.out.println("CV folds: " + CV_FOLDS);

        System.out.println(
                "Original features: "
                        + (trainData.numAttributes() - 1)
        );

        J48 j48 = new J48();

        compareConfigurations(
                "J48",
                j48,
                trainData,
                noIndex,
                yesIndex,
                results
        );

        OneR oneR = new OneR();
        oneR.setMinBucketSize(6);

        compareConfigurations(
                "OneR",
                oneR,
                trainData,
                noIndex,
                yesIndex,
                results
        );

        RandomForest randomForest = new RandomForest();
        randomForest.setNumIterations(100);
        randomForest.setSeed(RANDOM_SEED);

        compareConfigurations(
                "Random Forest",
                randomForest,
                trainData,
                noIndex,
                yesIndex,
                results
        );

        saveResults(results);

        System.out.println();
        System.out.println(
                "All combined experiments completed!"
        );

        System.out.println(
                "Total configurations evaluated: "
                        + results.size()
        );

        System.out.println(
                "Results saved to: " + OUTPUT_PATH
        );
    }

    private static void compareConfigurations(
            String modelName,
            Classifier baseClassifier,
            Instances trainData,
            int noIndex,
            int yesIndex,
            List<ImprovementResult> results
    ) throws Exception {

        int originalFeatures =
                trainData.numAttributes() - 1;

        System.out.println();
        System.out.println("================================");
        System.out.println("MODEL: " + modelName);
        System.out.println("================================");

        // ========================================
        // 1. BASELINE
        // ========================================

        evaluateConfiguration(
                modelName,
                "Baseline - All Features",
                AbstractClassifier.makeCopy(baseClassifier),
                originalFeatures,
                trainData,
                yesIndex,
                results
        );

        // ========================================
        // 2. FEATURE SELECTION ONLY
        // ========================================

        for (int k : TOP_K) {

            if (k > originalFeatures) {
                continue;
            }

            Classifier selectedModel =
                    FeatureSelectionPipeline.createInfoGainModel(
                            AbstractClassifier.makeCopy(baseClassifier),
                            k
                    );

            evaluateConfiguration(
                    modelName,
                    "InfoGain Top " + k,
                    selectedModel,
                    k,
                    trainData,
                    yesIndex,
                    results
            );
        }

        // ========================================
        // 3. CLASS WEIGHTING ONLY
        // ========================================

        for (int weight : YES_WEIGHTS) {

            Classifier weightedModel =
                    ClassWeightingConfig.createWeightedClassifier(
                            AbstractClassifier.makeCopy(baseClassifier),
                            noIndex,
                            yesIndex,
                            weight
                    );

            evaluateConfiguration(
                    modelName,
                    "Class Weighting YesWeight=" + weight,
                    weightedModel,
                    originalFeatures,
                    trainData,
                    yesIndex,
                    results
            );
        }

        // ========================================
        // 4. COMBINED IMPROVEMENT
        // ========================================

        for (int k : TOP_K) {

            if (k > originalFeatures) {
                continue;
            }

            for (int weight : YES_WEIGHTS) {

                // Build the weighted classifier.
                Classifier weightedModel =
                        ClassWeightingConfig.createWeightedClassifier(
                                AbstractClassifier.makeCopy(baseClassifier),
                                noIndex,
                                yesIndex,
                                weight
                        );

                // Feature Selection is the outer pipeline.
                // It selects attributes using training-fold data.
                //
                // The inner classifier then applies class
                // weights to its training-fold data.
                Classifier combinedModel =
                        FeatureSelectionPipeline.createInfoGainModel(
                                weightedModel,
                                k
                        );

                evaluateConfiguration(
                        modelName,
                        "InfoGain Top " + k
                                + " + YesWeight=" + weight,
                        combinedModel,
                        k,
                        trainData,
                        yesIndex,
                        results
                );
            }
        }
    }

    private static void evaluateConfiguration(
            String modelName,
            String methodName,
            Classifier model,
            int featureCount,
            Instances trainData,
            int positiveClass,
            List<ImprovementResult> results
    ) throws Exception {

        Evaluation evaluation = new Evaluation(trainData);

        evaluation.crossValidateModel(
                model,
                trainData,
                CV_FOLDS,
                new Random(RANDOM_SEED)
        );

        ImprovementResult result =
                ImprovementResult.fromEvaluation(
                        modelName,
                        methodName,
                        evaluation,
                        positiveClass
                );

        result.setFeatureCount(featureCount);
        results.add(result);

        System.out.println();
        System.out.println("--- " + methodName + " ---");

        System.out.printf(
                Locale.US,
                "Accuracy: %.2f%%%n",
                result.getAccuracy()
        );

        System.out.println(
                "Precision (Yes): "
                        + formatScore(result.getPrecision())
        );

        System.out.println(
                "Recall (Yes): "
                        + formatScore(result.getRecall())
        );

        System.out.println(
                "F1-score (Yes): "
                        + formatScore(result.getF1Score())
        );

        System.out.println(
                "ROC-AUC (Yes): "
                        + formatScore(result.getRocAuc())
        );
    }

    private static String formatScore(double value) {

        if (!Double.isFinite(value)) {
            return "N/A";
        }

        return String.format(
                Locale.US,
                "%.4f",
                value
        );
    }

    private static void saveResults(
            List<ImprovementResult> results
    ) throws Exception {

        Path outputFile = Path.of(OUTPUT_PATH);

        Files.createDirectories(
                outputFile.getParent()
        );

        StringBuilder csv = new StringBuilder();

        csv.append(ImprovementResult.csvHeader())
                .append(System.lineSeparator());

        for (ImprovementResult result : results) {
            csv.append(result.toCsvRow())
                    .append(System.lineSeparator());
        }

        Files.writeString(
                outputFile,
                csv.toString(),
                StandardCharsets.UTF_8
        );

        System.out.println();
        System.out.println("================================");
        System.out.println("COMBINED IMPROVEMENT RESULTS EXPORTED");
        System.out.println("================================");

        System.out.println(
                "File: " + outputFile.toAbsolutePath()
        );

        System.out.println(
                "Total results: " + results.size()
        );
    }
}
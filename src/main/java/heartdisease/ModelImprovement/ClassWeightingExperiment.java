package heartdisease.ModelImprovement;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

// Classification B 
import heartdisease.ClassificationB.NaiveBayesClassifier;
import heartdisease.ClassificationB.SimpleLogisticClassifier;

// Weka
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassWeightingExperiment {

    // ============================================
    // STEP 3: EXPERIMENT CONFIGURATION
    // ============================================

    private static final String DATA_PATH =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    private static final String OUTPUT_PATH =
            "results/step3_class_weighting_cv.csv";

    private static final int RANDOM_SEED = 42;
    private static final int CV_FOLDS = 10;

    // ============================================
    // MAIN METHOD
    // ============================================

    public static void main(String[] args) throws Exception {

        // Store all experiment results.
        List<ImprovementResult> results =
                new ArrayList<>();

        // ========================================
        // STEP 1: LOAD DATASET
        // ========================================

        Instances data = DataSource.read(DATA_PATH);

        if (data == null || data.numInstances() == 0) {
            throw new IllegalStateException(
                    "Cannot load dataset: " + DATA_PATH
            );
        }

        // Check target attribute.
        if (data.attribute(TARGET) == null) {
            throw new IllegalStateException(
                    "Target attribute not found: " + TARGET
            );
        }

        data.setClassIndex(
                data.attribute(TARGET).index()
        );

        // Check binary classification.
        if (!data.classAttribute().isNominal()
                || data.classAttribute().numValues() != 2) {

            throw new IllegalStateException(
                    "Expected a binary nominal target."
            );
        }

        // Define class indices.
        int noIndex =
                data.classAttribute().indexOfValue("No");

        int yesIndex =
                data.classAttribute().indexOfValue("Yes");

        if (noIndex < 0 || yesIndex < 0) {
            throw new IllegalStateException(
                    "Target must contain No and Yes."
            );
        }

        // ========================================
        // STEP 2: PREPARE TRAINING DATA
        // ========================================

        // Reproduce the Step 2 split.
        data.randomize(new Random(RANDOM_SEED));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);

        // IMPORTANT:
        // The held-out 2,000 test instances are
        // not used for selecting configurations.

        // ========================================
        // STEP 3: PRINT EXPERIMENT INFORMATION
        // ========================================

        System.out.println("================================");
        System.out.println("STEP 3 - MODEL IMPROVEMENT");
        System.out.println("DIRECT CLASS WEIGHTING EXPERIMENT");
        System.out.println("================================");

        System.out.println(
                "Training instances: "
                        + trainData.numInstances()
        );

        System.out.println(
                "CV folds: " + CV_FOLDS
        );

        System.out.println(
                "Original features: "
                        + (trainData.numAttributes() - 1)
        );

        System.out.println(
                "Total classifiers: 5"
        );

        // ========================================
        // STEP 4: J48 EXPERIMENTS
        // ========================================

        J48 j48 = new J48();

        compareWeights(
                "J48",
                j48,
                trainData,
                noIndex,
                yesIndex,
                results
        );

        // ========================================
        // STEP 5: ONER EXPERIMENTS
        // ========================================

        OneR oneR = new OneR();
        oneR.setMinBucketSize(6);

        compareWeights(
                "OneR",
                oneR,
                trainData,
                noIndex,
                yesIndex,
                results
        );

        // ========================================
        // STEP 6: RANDOM FOREST EXPERIMENTS
        // ========================================

        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setSeed(RANDOM_SEED);

        compareWeights(
                "Random Forest",
                rf,
                trainData,
                noIndex,
                yesIndex,
                results
        );

        // ========================================
        // STEP 7: NAIVE BAYES EXPERIMENTS
        // ========================================

        NaiveBayesClassifier naiveBayes =
                new NaiveBayesClassifier();

        // Use the original Weka classifier.
        // Cross-validation trains a fresh model
        // in each training fold.
        compareWeights(
                "Naive Bayes",
                naiveBayes.getClassifier(),
                trainData,
                noIndex,
                yesIndex,
                results
        );

        // ========================================
        // STEP 8: SIMPLE LOGISTIC EXPERIMENTS
        // ========================================

        SimpleLogisticClassifier simpleLogistic =
                new SimpleLogisticClassifier();

        // Keep Step 2's default configuration.
        // Do not train before cross-validation.
        compareWeights(
                "SimpleLogistic",
                simpleLogistic.getClassifier(),
                trainData,
                noIndex,
                yesIndex,
                results
        );

        // ========================================
        // STEP 9: EXPORT EXPERIMENT RESULTS
        // ========================================

        saveResultsToCsv(results);

        System.out.println();

        System.out.println(
                "All class weighting experiments completed!"
        );

        System.out.println(
                "Total configurations evaluated: "
                        + results.size()
        );

        System.out.println(
                "Results saved to: " + OUTPUT_PATH
        );
    }

    // ============================================
    // COMPARE CLASS WEIGHTING CONFIGURATIONS
    // ============================================

    private static void compareWeights(
            String modelName,
            Classifier baseClassifier,
            Instances trainData,
            int noIndex,
            int yesIndex,
            List<ImprovementResult> results
    ) throws Exception {

        System.out.println();
        System.out.println("================================");
        System.out.println("MODEL: " + modelName);
        System.out.println("================================");

        int featureCount =
                trainData.numAttributes() - 1;

        // ========================================
        // EXPERIMENT 0: BASELINE
        // ========================================

        // Original classifier without additional
        // class weighting.
        Classifier baseline =
                AbstractClassifier.makeCopy(
                        baseClassifier
                );

        evaluateCV(
                modelName,
                "Baseline - All Features",
                baseline,
                trainData,
                yesIndex,
                featureCount,
                results
        );

        // ========================================
        // EXPERIMENTS 1-3: DIRECT CLASS WEIGHTING
        // ========================================

        // ClassWeightingConfig supplies weights 2, 4, 8.
        for (int yesWeight :
                ClassWeightingConfig.getYesClassWeights()) {

            // Fresh classifier copy for each experiment.
            Classifier copiedClassifier =
                    AbstractClassifier.makeCopy(
                            baseClassifier
                    );

            // Apply direct class weighting.
            Classifier weightedModel =
                    ClassWeightingConfig.createWeightedClassifier(
                            copiedClassifier,
                            noIndex,
                            yesIndex,
                            yesWeight
                    );

            evaluateCV(
                    modelName,
                    "Class Weighting YesWeight=" + yesWeight,
                    weightedModel,
                    trainData,
                    yesIndex,
                    featureCount,
                    results
            );
        }
    }

    // ============================================
    // RUN CROSS-VALIDATION EXPERIMENT
    // ============================================

    private static void evaluateCV(
            String modelName,
            String methodName,
            Classifier model,
            Instances trainData,
            int positiveClass,
            int featureCount,
            List<ImprovementResult> results
    ) throws Exception {

        Evaluation evaluation =
                new Evaluation(trainData);

        // Stratified 10-fold cross-validation.
        //
        // Direct class weighting is performed
        // inside classifier training.
        //
        // Each fold uses its own training subset.
        // Validation data is not reweighted.
        evaluation.crossValidateModel(
                model,
                trainData,
                CV_FOLDS,
                new Random(RANDOM_SEED)
        );

        // ========================================
        // STORE EXPERIMENT RESULT
        // ========================================

        ImprovementResult result =
                ImprovementResult.fromEvaluation(
                        modelName,
                        methodName,
                        evaluation,
                        positiveClass
                );

        result.setFeatureCount(featureCount);
        results.add(result);

        // ========================================
        // PRINT EXPERIMENT RESULT
        // ========================================

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

    // ============================================
    // FORMAT METRIC VALUES
    // ============================================

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

    // ============================================
    // EXPORT STEP 3 RESULTS TO CSV
    // ============================================

    private static void saveResultsToCsv(
            List<ImprovementResult> results
    ) throws Exception {

        Path outputFile = Path.of(OUTPUT_PATH);

        // Create results directory if needed.
        Files.createDirectories(
                outputFile.getParent()
        );

        StringBuilder csv = new StringBuilder();

        // Write CSV header.
        csv.append(
                ImprovementResult.csvHeader()
        ).append(
                System.lineSeparator()
        );

        // Write experiment results.
        for (ImprovementResult result : results) {

            csv.append(
                    result.toCsvRow()
            ).append(
                    System.lineSeparator()
            );
        }

        // Save complete results.
        // Existing CSV is replaced by the new one.
        Files.writeString(
                outputFile,
                csv.toString(),
                StandardCharsets.UTF_8
        );

        System.out.println();
        System.out.println("================================");
        System.out.println("CLASS WEIGHTING RESULTS EXPORTED");
        System.out.println("================================");

        System.out.println(
                "File: " + outputFile.toAbsolutePath()
        );

        System.out.println(
                "Total results: " + results.size()
        );
    }
}

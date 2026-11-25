
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

public class CombinedImprovementExperiment {

    // ============================================
    // STEP 3: EXPERIMENT CONFIGURATION
    // ============================================

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

    // ============================================
    // MAIN METHOD
    // ============================================

    public static void main(String[] args) throws Exception {

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

        // Validate binary classification.
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

        // ========================================
        // STEP 2: PREPARE TRAINING DATA
        // ========================================

        // Reproduce the Step 2 train/test split.
        data.randomize(new Random(RANDOM_SEED));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);

        // IMPORTANT:
        // The held-out test set is not used
        // during Step 3 configuration selection.

        // ========================================
        // STEP 3: PRINT EXPERIMENT INFORMATION
        // ========================================

        System.out.println("================================");
        System.out.println("STEP 3 - COMBINED MODEL IMPROVEMENT");
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

        compareConfigurations(
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

        compareConfigurations(
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

        // ========================================
        // STEP 7: NAIVE BAYES EXPERIMENTS
        // ========================================

        NaiveBayesClassifier naiveBayes =
                new NaiveBayesClassifier();

        // Use Chau's original Weka classifier.
        // Do not train it before cross-validation.
        compareConfigurations(
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

        // Keep the original Step 2 configuration.
        // Training occurs inside each CV fold.
        compareConfigurations(
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

    // ============================================
    // COMPARE IMPROVEMENT CONFIGURATIONS
    // ============================================

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
        // EXPERIMENT 0: BASELINE
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
        // EXPERIMENTS 1-2: FEATURE SELECTION ONLY
        // ========================================

        for (int k : TOP_K) {

            if (k > originalFeatures) {
                System.out.println(
                        "Skipping InfoGain Top " + k
                                + ": too many features."
                );
                continue;
            }

            Classifier selectedModel =
                    FeatureSelectionPipeline.createInfoGainModel(
                            AbstractClassifier.makeCopy(
                                    baseClassifier
                            ),
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
        // EXPERIMENTS 3-4: CLASS WEIGHTING ONLY
        // ========================================

        for (int weight : YES_WEIGHTS) {

            Classifier weightedModel =
                    ClassWeightingConfig.createWeightedClassifier(
                            AbstractClassifier.makeCopy(
                                    baseClassifier
                            ),
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
        // EXPERIMENTS 5-8: COMBINED IMPROVEMENT
        // ========================================

        for (int k : TOP_K) {

            if (k > originalFeatures) {
                continue;
            }

            for (int weight : YES_WEIGHTS) {

                // Create a fresh weighted classifier.
                Classifier weightedModel =
                        ClassWeightingConfig.createWeightedClassifier(
                                AbstractClassifier.makeCopy(
                                        baseClassifier
                                ),
                                noIndex,
                                yesIndex,
                                weight
                        );

                // Feature Selection is the outer pipeline.
                //
                // InfoGain selects features from the
                // training portion of each CV fold.
                //
                // The inner weighted classifier then
                // increases the training weight of
                // instances belonging to class Yes.
                //
                // Validation instances are not used
                // to fit the feature selection filter
                // or to apply training weights.
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

    // ============================================
    // RUN CROSS-VALIDATION EXPERIMENT
    // ============================================

    private static void evaluateConfiguration(
            String modelName,
            String methodName,
            Classifier model,
            int featureCount,
            Instances trainData,
            int positiveClass,
            List<ImprovementResult> results
    ) throws Exception {

        Evaluation evaluation =
                new Evaluation(trainData);

        // Stratified 10-fold cross-validation.
        //
        // Every experiment uses the same random seed.
        // Feature selection and class weighting are
        // performed during model training.
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
    // EXPORT RESULTS TO CSV
    // ============================================

    private static void saveResults(
            List<ImprovementResult> results
    ) throws Exception {

        Path outputFile = Path.of(OUTPUT_PATH);

        // Create results directory if necessary.
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

        // Write all experiment results.
        for (ImprovementResult result : results) {

            csv.append(
                    result.toCsvRow()
            ).append(
                    System.lineSeparator()
            );
        }

        // Save the complete experiment results.
        // This overwrites the previous CSV.
        Files.writeString(
                outputFile,
                csv.toString(),
                StandardCharsets.UTF_8
        );

        System.out.println();
        System.out.println("================================");
        System.out.println(
                "COMBINED IMPROVEMENT RESULTS EXPORTED"
        );
        System.out.println("================================");

        System.out.println(
                "File: " + outputFile.toAbsolutePath()
        );

        System.out.println(
                "Total results: " + results.size()
        );
    }
}

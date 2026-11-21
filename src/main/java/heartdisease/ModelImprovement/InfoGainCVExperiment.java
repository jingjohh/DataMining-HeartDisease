
package heartdisease.ModelImprovement;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class InfoGainCVExperiment {

    // ============================================
    // STEP 3: EXPERIMENT CONFIGURATION
    // ============================================

    private static final String DATA_PATH =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    private static final String OUTPUT_PATH =
            "results/step3_feature_selection_cv.csv";

    private static final int CV_FOLDS = 10;
    private static final int RANDOM_SEED = 42;

    private static final int[] TOP_K =
            FeatureSelectionConfig.getTopKValues();

    // ============================================
    // MAIN METHOD
    // ============================================

    public static void main(String[] args) throws Exception {

        // Store all Step 3 experiment results.
        List<ImprovementResult> results = new ArrayList<>();

        // STEP 1: Load dataset.
        Instances data = DataSource.read(DATA_PATH);

        if (data == null) {
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

        // Define the positive class.
        int positiveClass =
                data.classAttribute().indexOfValue("Yes");

        if (positiveClass < 0) {
            throw new IllegalStateException(
                    "Positive class 'Yes' not found."
            );
        }

        // STEP 2: Reproduce the original train/test split.
        data.randomize(new Random(RANDOM_SEED));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);

        // IMPORTANT:
        // The remaining 2,000 test instances are not used
        // when selecting Step 3 configurations.

        // STEP 3: Print experiment information.
        System.out.println("================================");
        System.out.println("STEP 3 - MODEL IMPROVEMENT");
        System.out.println("INFORMATION GAIN EXPERIMENT");
        System.out.println("================================");

        System.out.println(
                "Training instances: " + trainData.numInstances()
        );

        System.out.println("CV folds: " + CV_FOLDS);

        System.out.println(
                "Original features: "
                        + (trainData.numAttributes() - 1)
        );

        // ============================================
        // STEP 4: J48 EXPERIMENTS
        // ============================================

        J48 j48 = new J48();

        compareMethods(
                "J48",
                j48,
                trainData,
                positiveClass,
                results
        );

        // ============================================
        // STEP 5: ONER EXPERIMENTS
        // ============================================

        OneR oneR = new OneR();
        oneR.setMinBucketSize(6);

        compareMethods(
                "OneR",
                oneR,
                trainData,
                positiveClass,
                results
        );

        // ============================================
        // STEP 6: RANDOM FOREST EXPERIMENTS
        // ============================================

        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);
        rf.setSeed(RANDOM_SEED);

        compareMethods(
                "Random Forest",
                rf,
                trainData,
                positiveClass,
                results
        );

        // ============================================
        // STEP 7: EXPORT EXPERIMENT RESULTS
        // ============================================

        saveResultsToCsv(results);

        System.out.println(
                "\nAll improvement experiments completed!"
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
    // COMPARE FEATURE SELECTION METHODS
    // ============================================

    private static void compareMethods(
            String name,
            Classifier baseClassifier,
            Instances trainData,
            int positiveClass,
            List<ImprovementResult> results
    ) throws Exception {

        System.out.println("\n================================");
        System.out.println("MODEL: " + name);
        System.out.println("================================");

        int originalFeatureCount =
                trainData.numAttributes() - 1;

        // ========================================
        // EXPERIMENT 0: BASELINE
        // ========================================

        Classifier baseline =
                AbstractClassifier.makeCopy(baseClassifier);

        evaluateCV(
                name,
                "Baseline - All Features",
                originalFeatureCount,
                baseline,
                trainData,
                positiveClass,
                results
        );

        // ========================================
        // EXPERIMENT 1: CFS + BESTFIRST
        // ========================================

        FilteredClassifier cfs =
                FeatureSelectionPipeline.createCFSModel(
                        AbstractClassifier.makeCopy(baseClassifier)
                );

        // -1 means that the exact feature count is
        // not fixed across cross-validation folds.
        evaluateCV(
                name,
                "CFS + BestFirst",
                -1,
                cfs,
                trainData,
                positiveClass,
                results
        );

        // ========================================
        // EXPERIMENTS 2-4: INFORMATION GAIN
        // ========================================

        for (int k : TOP_K) {

            if (k > originalFeatureCount) {
                System.out.println(
                        "Skipping InfoGain Top " + k
                                + ": too many features."
                );
                continue;
            }

            FilteredClassifier improved =
                    FeatureSelectionPipeline.createInfoGainModel(
                            AbstractClassifier.makeCopy(baseClassifier),
                            k
                    );

            evaluateCV(
                    name,
                    "InfoGain Top " + k,
                    k,
                    improved,
                    trainData,
                    positiveClass,
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
            int featureCount,
            Classifier model,
            Instances trainData,
            int positiveClass,
            List<ImprovementResult> results
    ) throws Exception {

        Evaluation evaluation = new Evaluation(trainData);

        // Each experiment uses identical folds and seed.
        //
        // When a FilteredClassifier is provided,
        // feature selection is fitted inside each
        // training fold to avoid data leakage.
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

        System.out.println("\n--- " + methodName + " ---");

        System.out.printf(
                "Accuracy: %.2f%%%n",
                result.getAccuracy()
        );

        System.out.printf(
                "Precision (Yes): %.4f%n",
                result.getPrecision()
        );

        System.out.printf(
                "Recall (Yes): %.4f%n",
                result.getRecall()
        );

        System.out.printf(
                "F1-score (Yes): %.4f%n",
                result.getF1Score()
        );

        System.out.printf(
                "ROC-AUC (Yes): %.4f%n",
                result.getRocAuc()
        );
    }

    // ============================================
    // EXPORT STEP 3 RESULTS TO CSV
    // ============================================

    private static void saveResultsToCsv(
            List<ImprovementResult> results
    ) throws Exception {

        Path outputFile = Path.of(OUTPUT_PATH);

        // Create the results folder if necessary.
        Files.createDirectories(outputFile.getParent());

        StringBuilder csv = new StringBuilder();

        // Write CSV header.
        csv.append(ImprovementResult.csvHeader())
                .append(System.lineSeparator());

        // Write each experiment result.
        for (ImprovementResult result : results) {
            csv.append(result.toCsvRow())
                    .append(System.lineSeparator());
        }

        // Overwrite the previous CSV with the latest
        // complete experiment results.
        Files.writeString(
                outputFile,
                csv.toString(),
                StandardCharsets.UTF_8
        );

        System.out.println(
                "\n================================"
        );

        System.out.println(
                "STEP 3 RESULTS EXPORTED"
        );

        System.out.println(
                "================================"
        );

        System.out.println(
                "File: " + outputFile.toAbsolutePath()
        );

        System.out.println(
                "Total results: " + results.size()
        );
    }
}

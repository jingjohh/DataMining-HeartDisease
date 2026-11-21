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

public class ClassWeightingExperiment {

    private static final String DATA_PATH =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    private static final String OUTPUT_PATH =
            "results/step3_class_weighting_cv.csv";

    private static final int RANDOM_SEED = 42;
    private static final int CV_FOLDS = 10;

    public static void main(String[] args) throws Exception {

        List<ImprovementResult> results = new ArrayList<>();

        // Load dataset.
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

        // Reproduce the Step 2 split.
        data.randomize(new Random(RANDOM_SEED));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);

        // Do not use the held-out test instances
        // for selecting Step 3 configurations.

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

        // J48
        J48 j48 = new J48();

        compareWeights(
                "J48",
                j48,
                trainData,
                noIndex,
                yesIndex,
                results
        );

        // OneR
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

        // Random Forest
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

        // Baseline: original model, all features.
        Classifier baseline =
                AbstractClassifier.makeCopy(baseClassifier);

        evaluateCV(
                modelName,
                "Baseline - All Features",
                baseline,
                trainData,
                yesIndex,
                featureCount,
                results
        );

        // Direct class weighting configurations.
        for (int yesWeight :
                ClassWeightingConfig.getYesClassWeights()) {

            Classifier copiedClassifier =
                    AbstractClassifier.makeCopy(
                            baseClassifier
                    );

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

    private static void saveResultsToCsv(
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
        System.out.println(
                "================================"
        );
        System.out.println(
                "CLASS WEIGHTING RESULTS EXPORTED"
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
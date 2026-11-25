
package heartdisease.Evaluation;

import java.util.Locale;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

public class ModelEvaluator {

    private static final int DEFAULT_FOLDS = 10;
    private static final int RANDOM_SEED = 42;

    // ==========================================
    // RESULT CONTAINER
    // ==========================================

    public static class TimedResult {

        private final Evaluation evaluation;
        private final double trainingMs;
        private final double predictionMs;
        private final int predictedInstances;
        private final Classifier trainedModel;

        public TimedResult(
                Evaluation evaluation,
                long trainingNanos,
                long predictionNanos,
                int predictedInstances,
                Classifier trainedModel) {

            this.evaluation = evaluation;
            this.trainingMs = trainingNanos / 1_000_000.0;
            this.predictionMs = predictionNanos / 1_000_000.0;
            this.predictedInstances = predictedInstances;
            this.trainedModel = trainedModel;
        }

        public Evaluation getEvaluation() {
            return evaluation;
        }

        public double getTrainingMs() {
            return trainingMs;
        }

        public double getPredictionMs() {
            return predictionMs;
        }

        public int getPredictedInstances() {
            return predictedInstances;
        }

        public Classifier getTrainedModel() {
            return trainedModel;
        }
    }

    // ==========================================
    // DATA VALIDATION
    // ==========================================

    private static void validateData(
            Classifier classifier,
            Instances data) {

        if (classifier == null) {
            throw new IllegalArgumentException(
                "Classifier cannot be null."
            );
        }

        if (data == null || data.numInstances() == 0) {
            throw new IllegalArgumentException(
                "Dataset cannot be null or empty."
            );
        }

        if (data.classIndex() < 0 ||
                !data.classAttribute().isNominal() ||
                data.classAttribute().numValues() != 2) {

            throw new IllegalArgumentException(
                "A binary nominal target is required."
            );
        }
    }

    // ==========================================
    // 10-FOLD CROSS-VALIDATION
    // ==========================================

    // Retain backward compatibility
    public static Evaluation evaluateCV(
            Classifier classifier,
            Instances trainData) throws Exception {

        return evaluateCVWithTiming(
            classifier, trainData
        ).getEvaluation();
    }

    public static TimedResult evaluateCVWithTiming(
            Classifier classifier,
            Instances trainData) throws Exception {

        validateData(classifier, trainData);

        if (trainData.numInstances() < DEFAULT_FOLDS) {
            throw new IllegalArgumentException(
                "Not enough instances for 10-fold CV."
            );
        }

        // Copy the dataset to avoid modifying original data
        Instances shuffled = new Instances(trainData);

        Random random = new Random(RANDOM_SEED);
        shuffled.randomize(random);
        shuffled.stratify(DEFAULT_FOLDS);

        Evaluation evaluation = new Evaluation(shuffled);

        long totalTrainingNanos = 0;
        long totalPredictionNanos = 0;
        int predictionCount = 0;

        // Run each fold independently
        for (int fold = 0; fold < DEFAULT_FOLDS; fold++) {

            Instances foldTrain = shuffled.trainCV(
                DEFAULT_FOLDS, fold, random
            );

            Instances foldTest = shuffled.testCV(
                DEFAULT_FOLDS, fold
            );

            // Fresh classifier for each fold
            Classifier foldModel =
                AbstractClassifier.makeCopy(classifier);

            // Measure training time
            long start = System.nanoTime();

            foldModel.buildClassifier(foldTrain);

            totalTrainingNanos +=
                System.nanoTime() - start;

            // Evaluate predictions from this fold
            for (int i = 0; i < foldTest.numInstances(); i++) {

                Instance instance = foldTest.instance(i);

                // Measure prediction only
                start = System.nanoTime();

                double[] distribution =
                    foldModel.distributionForInstance(instance);

                totalPredictionNanos +=
                    System.nanoTime() - start;

                // Record prediction for Weka metrics
                evaluation.evaluateModelOnceAndRecordPrediction(
                    distribution, instance
                );

                predictionCount++;
            }
        }

        return new TimedResult(
            evaluation,
            totalTrainingNanos,
            totalPredictionNanos,
            predictionCount,
            null
        );
    }

    // ==========================================
    // HOLD-OUT TEST EVALUATION
    // ==========================================

    public static TimedResult evaluateHoldoutWithTiming(
            Classifier classifier,
            Instances trainData,
            Instances testData) throws Exception {

        validateData(classifier, trainData);
        validateData(classifier, testData);

        // Training and test headers must match
        String mismatch = trainData.equalHeadersMsg(testData);

        if (mismatch != null) {
            throw new IllegalArgumentException(
                "Incompatible train/test datasets: " + mismatch
            );
        }

        // Create a fresh model
        Classifier trainedModel =
            AbstractClassifier.makeCopy(classifier);

        // Train using training data only
        long start = System.nanoTime();

        trainedModel.buildClassifier(trainData);

        long trainingNanos = System.nanoTime() - start;

        Evaluation evaluation = new Evaluation(trainData);

        long predictionNanos = 0;

        // Predict the independent test set
        for (int i = 0; i < testData.numInstances(); i++) {

            Instance instance = testData.instance(i);

            start = System.nanoTime();

            double[] distribution =
                trainedModel.distributionForInstance(instance);

            predictionNanos += System.nanoTime() - start;

            evaluation.evaluateModelOnceAndRecordPrediction(
                distribution, instance
            );
        }

        return new TimedResult(
            evaluation,
            trainingNanos,
            predictionNanos,
            testData.numInstances(),
            trainedModel
        );
    }

    // ==========================================
    // PRINT EVALUATION METRICS
    // ==========================================

    private static String formatMetric(double value) {
        return Double.isFinite(value)
            ? String.format(Locale.US, "%.4f", value)
            : "N/A";
    }

    public static void printResults(
            String modelName,
            Evaluation evaluation,
            Instances data) throws Exception {

        int positiveClass =
            data.classAttribute().indexOfValue("Yes");

        if (positiveClass < 0) {
            throw new IllegalArgumentException(
                "Positive class 'Yes' not found."
            );
        }

        System.out.println(
            "\n===== " + modelName + " ====="
        );

        System.out.printf(
            Locale.US, "Accuracy: %.2f%%%n",
            evaluation.pctCorrect()
        );

        System.out.println(
            "Precision (Yes): " +
            formatMetric(evaluation.precision(positiveClass))
        );

        System.out.println(
            "Recall (Yes): " +
            formatMetric(evaluation.recall(positiveClass))
        );

        System.out.println(
            "F1-score (Yes): " +
            formatMetric(evaluation.fMeasure(positiveClass))
        );

        System.out.println(
            "ROC-AUC (Yes): " +
            formatMetric(evaluation.areaUnderROC(positiveClass))
        );

        System.out.println(evaluation.toMatrixString());
    }

    // ==========================================
    // PRINT METRICS AND RUNTIME
    // ==========================================

    public static void printTimedResults(
            String modelName,
            TimedResult result,
            Instances data) throws Exception {

        printResults(
            modelName,
            result.getEvaluation(),
            data
        );

        System.out.println("--- Runtime ---");

        System.out.printf(
            Locale.US,
            "Training time: %.3f ms%n",
            result.getTrainingMs()
        );

        System.out.printf(
            Locale.US,
            "Prediction time: %.3f ms%n",
            result.getPredictionMs()
        );

        System.out.printf(
            Locale.US,
            "Average prediction: %.6f ms/instance%n",
            result.getPredictionMs()
                / result.getPredictedInstances()
        );
    }
}

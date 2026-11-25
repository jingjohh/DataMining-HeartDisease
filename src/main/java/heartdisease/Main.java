
package heartdisease;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

import heartdisease.ClassificationA.J48Classifier;
import heartdisease.ClassificationA.OneRClassifier;
import heartdisease.ClassificationA.RandomForestClassifier;
import heartdisease.ClassificationB.NaiveBayesClassifier;
import heartdisease.ClassificationB.SimpleLogisticClassifier;
import heartdisease.Evaluation.ModelComparator;
import heartdisease.Evaluation.ModelEvaluator;
import heartdisease.ModelImprovement.ClassWeightingConfig;
import heartdisease.ModelImprovement.FeatureSelectionConfig;
import heartdisease.ModelImprovement.FeatureSelectionPipeline;
import heartdisease.ModelImprovement.ImprovementResult;
import heartdisease.Preprocessing.DataLoader;
import heartdisease.Preprocessing.DataPreprocessing;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class Main {

    private static final String DEFAULT_DATA =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    private static final int SEED = 42;

    // Store one evaluated model configuration
    private record Run(
            String name,
            String method,
            Classifier prototype,
            ImprovementResult metrics,
            ModelEvaluator.TimedResult timing
    ) {}

    public static void main(String[] args) throws Exception {

        // ==========================================
        // STEP 0: PARSE COMMAND-LINE ARGUMENTS
        // ==========================================

        String dataPath = null;
        String trainPath = null;
        String testPath = null;
        boolean finalTest = false;

        for (int i = 0; i < args.length; i++) {

            String option = args[i];

            if ("--final-test".equals(option)) {
                finalTest = true;
                continue;
            }

            if (!option.equals("--data")
                    && !option.equals("--train")
                    && !option.equals("--test")) {

                throw new IllegalArgumentException(
                        "Unknown argument: " + option
                );
            }

            if (i + 1 >= args.length) {
                throw new IllegalArgumentException(
                        "Missing path after " + option
                );
            }

            String path = args[++i];

            switch (option) {
                case "--data" -> dataPath = path;
                case "--train" -> trainPath = path;
                case "--test" -> testPath = path;
                default -> throw new IllegalStateException();
            }
        }

        boolean externalSplit =
                trainPath != null || testPath != null;

        if (externalSplit &&
                (trainPath == null || testPath == null
                        || dataPath != null)) {

            throw new IllegalArgumentException(
                    "Use --data OR both --train and --test."
            );
        }

        // ==========================================
        // STEP 1: LOAD AND PREPARE DATA
        // ==========================================

        System.out.println(
                "\n===== STEP 1: DATA PREPROCESSING ====="
        );

        Instances trainData;
        Instances testData;

        if (externalSplit) {

            Path trainFile =
                    Path.of(trainPath).toRealPath();

            Path testFile =
                    Path.of(testPath).toRealPath();

            if (trainFile.equals(testFile)) {
                throw new IllegalArgumentException(
                        "Train and test must be different files."
                );
            }

            trainData = loadPrepared(trainPath);
            testData = loadPrepared(testPath);

        } else {

            String selectedPath = dataPath == null
                    ? DEFAULT_DATA : dataPath;

            Instances data = loadPrepared(selectedPath);

            // Match the original project's 80/20 split
            data.randomize(new Random(SEED));
            data.stratify(5);

            trainData = data.trainCV(5, 0);
            testData = data.testCV(5, 0);
        }

        String headerError =
                trainData.equalHeadersMsg(testData);

        if (headerError != null) {
            throw new IllegalArgumentException(
                    "Train/test schema mismatch: " + headerError
            );
        }

        int noIndex = trainData.classAttribute()
                .indexOfValue("No");

        int yesIndex = trainData.classAttribute()
                .indexOfValue("Yes");

        if (noIndex < 0 || yesIndex < 0) {
            throw new IllegalArgumentException(
                    "Target must contain No and Yes."
            );
        }

        System.out.println(
                "Training instances: "
                        + trainData.numInstances()
        );

        System.out.println(
                "Held-out test instances: "
                        + testData.numInstances()
        );

        // ==========================================
        // STEP 2: INITIALIZE CLASSIFIERS
        // ==========================================

        System.out.println(
                "\n===== STEP 2: CLASSIFICATION ====="
        );

        String[] names = {
                "J48",
                "OneR",
                "RandomForest",
                "NaiveBayes",
                "SimpleLogistic"
        };

        Classifier[] classifiers = {
                new J48Classifier().getModel(),
                new OneRClassifier().getModel(),
                new RandomForestClassifier().getModel(),
                new NaiveBayesClassifier().getClassifier(),
                new SimpleLogisticClassifier().getClassifier()
        };

        List<Run> allRuns = new ArrayList<>();
        List<Run> baselineRuns = new ArrayList<>();

        List<ModelComparator.Comparison> comparisons =
                new ArrayList<>();

        // ==========================================
        // STEP 3 & 4: IMPROVEMENT AND EVALUATION
        // ==========================================

        System.out.println(
                "\n===== STEPS 3-4: CV AND COMPARISON ====="
        );


        for (int i = 0; i < classifiers.length; i++) {

                String name = names[i];
                Classifier base = classifiers[i];
            
                System.out.println(
                    "\n===================================="
                );
                System.out.println("CLASSIFIER: " + name);
                System.out.println(
                    "===================================="
                );
            
                // ==========================================
                // EXPERIMENT 0: BASELINE
                // ==========================================
            
                Run baseline = evaluate(
                    name,
                    "Baseline - All Features",
                    withImputer(
                        AbstractClassifier.makeCopy(base)
                    ),
                    trainData,
                    yesIndex
                );
            
                allRuns.add(baseline);
                baselineRuns.add(baseline);
            
                // ==========================================
                // EXPERIMENT 1: CFS + BESTFIRST
                // ==========================================
            
                Classifier cfs =
                    FeatureSelectionPipeline.createCFSModel(
                        AbstractClassifier.makeCopy(base)
                    );
            
                evaluateImprovement(
                    name,
                    "CFS + BestFirst",
                    cfs,
                    trainData,
                    yesIndex,
                    baseline,
                    allRuns,
                    comparisons
                );
            
                // ==========================================
                // EXPERIMENTS 2-4: INFORMATION GAIN
                // ==========================================
            
                int originalFeatures =
                    trainData.numAttributes() - 1;
            
                for (int k :
                        FeatureSelectionConfig.getTopKValues()) {
            
                    if (k > originalFeatures) {
                        System.out.println(
                            "Skipping InfoGain Top " + k
                        );
                        continue;
                    }
            
                    Classifier infoGain =
                        FeatureSelectionPipeline.createInfoGainModel(
                            AbstractClassifier.makeCopy(base),
                            k
                        );
            
                    evaluateImprovement(
                        name,
                        "InfoGain Top " + k,
                        infoGain,
                        trainData,
                        yesIndex,
                        baseline,
                        allRuns,
                        comparisons
                    );
                }
            
                // ==========================================
                // EXPERIMENTS 5-7: CLASS WEIGHTING
                // ==========================================
            
                for (int weight :
                        ClassWeightingConfig.getYesClassWeights()) {
            
                    Classifier weighted =
                        ClassWeightingConfig.createWeightedClassifier(
                            AbstractClassifier.makeCopy(base),
                            noIndex,
                            yesIndex,
                            weight
                        );
            
                    evaluateImprovement(
                        name,
                        "Class Weighting YesWeight=" + weight,
                        weighted,
                        trainData,
                        yesIndex,
                        baseline,
                        allRuns,
                        comparisons
                    );
                }
            
                // ==========================================
                // EXPERIMENTS 8-11: COMBINED IMPROVEMENT
                // Same settings as Step 3 experiment
                // ==========================================
            
                int[] combinedTopK = {10, 15};
                int[] combinedWeights = {4, 8};
            
                for (int k : combinedTopK) {
            
                    if (k > originalFeatures) {
                        continue;
                    }
            
                    for (int weight : combinedWeights) {
            
                        // First create the weighted classifier
                        Classifier weighted =
                            ClassWeightingConfig.createWeightedClassifier(
                                AbstractClassifier.makeCopy(base),
                                noIndex,
                                yesIndex,
                                weight
                            );
            
                        // Then combine it with InfoGain
                        Classifier combined =
                            FeatureSelectionPipeline.createInfoGainModel(
                                weighted,
                                k
                            );
            
                        evaluateImprovement(
                            name,
                            "InfoGain Top " + k
                                + " + YesWeight=" + weight,
                            combined,
                            trainData,
                            yesIndex,
                            baseline,
                            allRuns,
                            comparisons
                        );
                    }
                }
            }

        System.out.println(
            "===== EXPERIMENT SUMMARY ====="
        );
            
        System.out.println(
            "\nTotal CV configurations: " + allRuns.size()
        );
            
        System.out.println(
            "Total comparison pairs: " + comparisons.size()
        );
            
        // ==========================================
        // EXPORT RESULTS
        // ==========================================

        ModelComparator.saveCsv(
                comparisons,
                "results/step4_model_comparison.csv"
        );

        saveRuntimeCsv(
                allRuns,
                "results/step4_cv_runtime.csv"
        );

        // ==========================================
        // MODEL SELECTION USING BALANCED ACCURACY
        // ==========================================

        Run best = selectBestModel(
                allRuns,
                noIndex,
                yesIndex
        );

        double[][] cm =
                best.metrics().getConfusionMatrix();

        double specificity =
                cm[noIndex][noIndex]
                / (cm[noIndex][noIndex]
                   + cm[noIndex][yesIndex]);

        double recall =
                cm[yesIndex][yesIndex]
                / (cm[yesIndex][yesIndex]
                   + cm[yesIndex][noIndex]);

        double balancedAccuracy =
                (specificity + recall) / 2.0;

        System.out.println(
                "\n===== SELECTED MODEL BY BALANCED ACCURACY ====="
        );

        System.out.println("Model: " + best.name());
        System.out.println("Method: " + best.method());

        System.out.printf(
                Locale.US,
                "Balanced Accuracy: %.4f%n",
                balancedAccuracy
        );

        System.out.printf(
                Locale.US,
                "Specificity (No): %.4f%n",
                specificity
        );

        System.out.printf(
                Locale.US,
                "Recall (Yes): %.4f%n",
                recall
        );

        System.out.printf(
                Locale.US,
                "F1 (Yes): %.4f%n",
                best.metrics().getF1Score()
        );

        System.out.printf(
                Locale.US,
                "ROC-AUC: %.4f%n",
                best.metrics().getRocAuc()
        );


        // ==========================================
        // TRAIN AND SAVE BASELINE MODELS
        // ==========================================

        Files.createDirectories(Path.of("models"));

        for (Run baseline : baselineRuns) {

            Classifier fitted =
                    AbstractClassifier.makeCopy(
                            baseline.prototype()
                    );

            fitted.buildClassifier(trainData);

            SerializationHelper.write(
                    "models/" + baseline.name() + ".model",
                    fitted
            );
        }

        // ==========================================
        // FINAL TEST: EXPLICIT OPT-IN ONLY
        // ==========================================

        Classifier selectedModel;

        if (finalTest) {

            ModelEvaluator.TimedResult finalResult =
                    ModelEvaluator.evaluateHoldoutWithTiming(
                            best.prototype(),
                            trainData,
                            testData
                    );

            ModelEvaluator.printTimedResults(
                    best.name() + " - Final Test",
                    finalResult,
                    testData
            );

            selectedModel = finalResult.getTrainedModel();

        } else {

            selectedModel =
                    AbstractClassifier.makeCopy(
                            best.prototype()
                    );

            selectedModel.buildClassifier(trainData);

            System.out.println(
                    "\nFinal test was NOT evaluated."
            );
        }

        // ==========================================
        // SAVE AND VERIFY SELECTED MODEL
        // ==========================================

        String modelFile = "models/"
                + best.name()
                + "_"
                + best.method().replaceAll(
                        "[^A-Za-z0-9]+", "_"
                )
                + ".model";

        SerializationHelper.write(
                modelFile,
                selectedModel
        );

        Classifier reloaded =
                (Classifier) SerializationHelper.read(
                        modelFile
                );

        // Verify prediction after serialization
        double before = selectedModel.classifyInstance(
                trainData.instance(0)
        );

        double after = reloaded.classifyInstance(
                trainData.instance(0)
        );

        if (Double.compare(before, after) != 0) {
            throw new IllegalStateException(
                    "Save/load verification failed."
            );
        }

        System.out.println(
                "Selected model saved: " + modelFile
        );

        System.out.println(
                "Save/load verification: PASS"
        );

        System.out.println(
                "\n===== FRAMEWORK COMPLETED ====="
        );
    }

    // ==========================================
    // DATA LOADING AND BASIC PREPARATION
    // ==========================================

    private static Instances loadPrepared(
            String path) throws Exception {

        Path file = Path.of(path).toAbsolutePath()
                .normalize();

        if (!Files.isRegularFile(file)) {
            throw new IllegalArgumentException(
                    "Dataset not found: " + file
            );
        }

        String name = file.getFileName()
                .toString().toLowerCase(Locale.ROOT);

        Instances data;

        if (name.endsWith(".csv")) {
            data = DataLoader.loadDataset(file.toString());

        } else if (name.endsWith(".arff")) {
            data = DataSource.read(file.toString());

        } else {
            throw new IllegalArgumentException(
                    "Only CSV and ARFF are supported."
            );
        }

        if (data == null || data.numInstances() == 0) {
            throw new IllegalArgumentException(
                    "Dataset is empty."
            );
        }

        if (data.attribute(TARGET) == null) {
            throw new IllegalArgumentException(
                    "Target not found: " + TARGET
            );
        }

        // Only removes rows with missing labels.
        // Does not fit imputation or scaling statistics.
        return DataPreprocessing.prepareSharedDataset(data);
    }

    // ==========================================
    // FOLD-SAFE MISSING VALUE IMPUTATION
    // ==========================================

    private static Classifier withImputer(
            Classifier classifier) {

        FilteredClassifier pipeline =
                new FilteredClassifier();

        pipeline.setFilter(new ReplaceMissingValues());
        pipeline.setClassifier(classifier);

        return pipeline;
    }

    // ==========================================
    // SHARED CROSS-VALIDATION RUNNER
    // ==========================================

    private static Run evaluate(
            String name,
            String method,
            Classifier classifier,
            Instances trainData,
            int positiveClass) throws Exception {

        ModelEvaluator.TimedResult timing =
                ModelEvaluator.evaluateCVWithTiming(
                        classifier,
                        trainData
                );

        ImprovementResult metrics =
                ImprovementResult.fromEvaluation(
                        name,
                        method,
                        timing.getEvaluation(),
                        positiveClass
                );

        System.out.printf(
                Locale.US,
                "%-16s | %-30s | "
                + "Accuracy: %.2f%% | Recall: %.4f | F1: %s%n",
                name,
                method,
                metrics.getAccuracy(),
                metrics.getRecall(),
                Double.isFinite(metrics.getF1Score())
                        ? String.format(Locale.US, "%.4f",
                                metrics.getF1Score())
                        : "N/A"
        );

        return new Run(
                name, method, classifier, metrics, timing
        );
    }


    // ==========================================
    // EVALUATE IMPROVEMENT
    // ==========================================

    private static void evaluateImprovement(
                String modelName,
                String methodName,
                Classifier improvedClassifier,
                Instances trainData,
                int positiveClass,
                Run baseline,
                List<Run> allRuns,
                List<ModelComparator.Comparison> comparisons
        ) throws Exception {

        // Apply fold-safe imputation
        Classifier pipeline = withImputer(
            improvedClassifier
        );

        // Evaluate the improved model
        Run improved = evaluate(
                modelName,
                methodName,
                pipeline,
                trainData,
                positiveClass
        );

        // Store the evaluation result
        allRuns.add(improved);

        // Compare with baseline
        ModelComparator.Comparison comparison =
                ModelComparator.compare(
                        baseline.metrics(),
                        improved.metrics()
                );

        comparisons.add(comparison);
    }

    // ==========================================
    // SELECT BEST MODEL
    // ==========================================

    private static Run selectBestModel(
        List<Run> runs,
        int noIndex,
        int yesIndex) {

    Run best = null;

    double bestScore = Double.NEGATIVE_INFINITY;
    double bestF1 = Double.NEGATIVE_INFINITY;

    for (Run run : runs) {

        double[][] cm =
                run.metrics().getConfusionMatrix();

        if (cm == null || cm.length != 2) {
            continue;
        }

        // Count predicted instances for each class
        double predictedNo =
                cm[noIndex][noIndex]
                + cm[yesIndex][noIndex];

        double predictedYes =
                cm[noIndex][yesIndex]
                + cm[yesIndex][yesIndex];

        // Exclude models predicting only one class
        if (predictedNo == 0 || predictedYes == 0) {
            continue;
        }

        // Count actual instances
        double actualNo =
                cm[noIndex][noIndex]
                + cm[noIndex][yesIndex];

        double actualYes =
                cm[yesIndex][noIndex]
                + cm[yesIndex][yesIndex];

        if (actualNo == 0 || actualYes == 0) {
            continue;
        }

        // Calculate balanced accuracy
        double specificity =
                cm[noIndex][noIndex] / actualNo;

        double recall =
                cm[yesIndex][yesIndex] / actualYes;

        double score =
                (specificity + recall) / 2.0;

        if (!Double.isFinite(score)) {
            continue;
        }

        // Use F1 as a tie-breaker
        double f1 = run.metrics().getF1Score();

        if (!Double.isFinite(f1)) {
            f1 = Double.NEGATIVE_INFINITY;
        }

        if (best == null
                || score > bestScore + 1e-9
                || (Math.abs(score - bestScore) <= 1e-9
                    && f1 > bestF1)) {

            best = run;
            bestScore = score;
            bestF1 = f1;
        }
    }

    if (best == null) {
        throw new IllegalStateException(
                "No model predicts both classes."
        );
    }

    return best;
}


    // ==========================================
    // EXPORT CROSS-VALIDATION RUNTIME
    // ==========================================

    private static void saveRuntimeCsv(
            List<Run> runs,
            String fileName) throws Exception {

        List<String> rows = new ArrayList<>();

        rows.add(
                "Model,Method,TrainingMs10Fold,"
                + "PredictionMs10Fold,MeanPredictionMs"
        );

        for (Run run : runs) {

            double trainMs = run.timing().getTrainingMs();
            double predictMs =
                    run.timing().getPredictionMs();

            double meanPredictionMs = predictMs
                    / run.timing().getPredictedInstances();

            rows.add(String.format(
                    Locale.US,
                    "\"%s\",\"%s\",%.6f,%.6f,%.9f",
                    run.name(),
                    run.method(),
                    trainMs,
                    predictMs,
                    meanPredictionMs
            ));
        }

        Path output = Path.of(fileName);

        if (output.getParent() != null) {
            Files.createDirectories(output.getParent());
        }

        Files.write(output, rows);

        System.out.println(
                "\nRuntime CSV saved: "
                        + output.toAbsolutePath()
        );
    }
}

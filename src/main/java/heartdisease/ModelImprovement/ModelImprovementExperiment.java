
package heartdisease.ModelImprovement;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.rules.OneR;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

public class ModelImprovementExperiment {

    private static final String DATA_PATH =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    public static void main(String[] args) throws Exception {

        // STEP 1: Load dataset
        Instances data = DataSource.read(DATA_PATH);

        if (data.attribute(TARGET) == null) {
            throw new IllegalStateException(
                "Target attribute not found: " + TARGET
            );
        }

        data.setClassIndex(data.attribute(TARGET).index());

        int positiveClass = data.classAttribute()
                .indexOfValue("Yes");

        if (positiveClass < 0) {
            throw new IllegalStateException(
                "Positive class 'Yes' not found!"
            );
        }

        // STEP 2: Same data split as ClassificationA
        data.randomize(new Random(42));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);
        Instances testData = data.testCV(5, 0);

        System.out.println("================================");
        System.out.println("STEP 3 - MODEL IMPROVEMENT");
        System.out.println("================================");

        System.out.println("Training: "
                + trainData.numInstances());

        System.out.println("Testing: "
                + testData.numInstances());

        System.out.println("Original features: "
                + (data.numAttributes() - 1));

        // STEP 3: Inspect selected features
        printSelectedFeatures(trainData);

        // STEP 4: Run comparisons
        compare("J48", createJ48(),
                trainData, testData, positiveClass);

        compare("OneR", createOneR(),
                trainData, testData, positiveClass);

        compare("Random Forest", createRandomForest(),
                trainData, testData, positiveClass);

        System.out.println(
            "\nAll improvement experiments completed!"
        );
    }

    // Classifier configurations
    private static J48 createJ48() {
        return new J48();
    }

    private static OneR createOneR() {
        OneR model = new OneR();
        model.setMinBucketSize(6);
        return model;
    }

    private static RandomForest createRandomForest() {
        RandomForest model = new RandomForest();
        model.setNumIterations(100);
        model.setSeed(42);
        return model;
    }

    // Inspect features using training data only
    private static void printSelectedFeatures(
            Instances trainData) throws Exception {

        AttributeSelection selector =
                new AttributeSelection();

        selector.setEvaluator(
                FeatureSelectionConfig.createEvaluator()
        );

        selector.setSearch(
                FeatureSelectionConfig.createSearchMethod()
        );

        selector.setInputFormat(trainData);

        Instances selected = Filter.useFilter(
                trainData, selector
        );

        System.out.println("\n=== SELECTED FEATURES ===");

        System.out.println("Selected feature count: "
                + (selected.numAttributes() - 1));

        for (int i = 0; i < selected.numAttributes(); i++) {

            if (i != selected.classIndex()) {
                System.out.println(
                    "- " + selected.attribute(i).name()
                );
            }
        }
    }

    // Compare baseline and improved versions
    private static void compare(
            String name,
            Classifier baseClassifier,
            Instances trainData,
            Instances testData,
            int positiveClass) throws Exception {

        System.out.println("\n================================");
        System.out.println("MODEL: " + name);
        System.out.println("================================");

        // BASELINE
        // Create an independent copy of the classifier
        Classifier baseline =
                AbstractClassifier.makeCopy(baseClassifier);

        baseline.buildClassifier(trainData);

        Evaluation before = new Evaluation(trainData);
        before.evaluateModel(baseline, testData);

        // IMPROVED
        // Create another independent classifier copy
        Classifier improvedBase =
                AbstractClassifier.makeCopy(baseClassifier);

        // Apply CFS + BestFirst Feature Selection
        FilteredClassifier improved =
                FeatureSelectionPipeline.createCFSModel(
                        improvedBase
                );

        improved.buildClassifier(trainData);

        Evaluation after = new Evaluation(trainData);
        after.evaluateModel(improved, testData);

        // PRINT RESULTS
        System.out.println("\n--- BEFORE IMPROVEMENT ---");
        printMetrics(before, positiveClass);

        System.out.println("\n--- AFTER IMPROVEMENT ---");
        printMetrics(after, positiveClass);

        // COMPARE
        System.out.println("\n--- CHANGES ---");

        System.out.printf("Accuracy change: %+.2f pp%n",
                after.pctCorrect() - before.pctCorrect());

        System.out.printf("F1 change: %+.4f%n",
                after.fMeasure(positiveClass)
                        - before.fMeasure(positiveClass));

        System.out.printf("Recall change: %+.4f%n",
                after.recall(positiveClass)
                        - before.recall(positiveClass));

        System.out.printf("ROC-AUC change: %+.4f%n",
                after.areaUnderROC(positiveClass)
                        - before.areaUnderROC(positiveClass));

        System.out.println("\n--- CONFUSION MATRIX: BEFORE ---");
        System.out.println(before.toMatrixString());

        System.out.println("\n--- CONFUSION MATRIX: AFTER ---");
        System.out.println(after.toMatrixString());
    }

    // Print evaluation metrics
    private static void printMetrics(
            Evaluation eval,
            int positiveClass) {

        System.out.printf("Accuracy: %.2f%%%n",
                eval.pctCorrect());

        System.out.printf("Precision (Yes): %.4f%n",
                eval.precision(positiveClass));

        System.out.printf("Recall (Yes): %.4f%n",
                eval.recall(positiveClass));

        System.out.printf("F1-score (Yes): %.4f%n",
                eval.fMeasure(positiveClass));

        System.out.printf("ROC-AUC (Yes): %.4f%n",
                eval.areaUnderROC(positiveClass));
    }
}

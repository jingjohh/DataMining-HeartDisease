
package heartdisease.Evaluation;

// Classification A
import java.util.Random;

import heartdisease.ClassificationA.J48Classifier;
import heartdisease.ClassificationA.OneRClassifier;
import heartdisease.ClassificationA.RandomForestClassifier;
import heartdisease.ClassificationB.NaiveBayesClassifier;
import heartdisease.ClassificationB.SimpleLogisticClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class EvaluationTest {

    private static final String DATA_PATH =
        "data/processed/heart_disease.arff";

    private static final String TARGET =
        "Heart Disease Status";

    private static final int SEED = 42;

    public static void main(String[] args) throws Exception {

        // ==========================================
        // STEP 1: LOAD DATASET
        // ==========================================

        Instances data = DataSource.read(DATA_PATH);

        if (data == null || data.numInstances() == 0) {
            throw new IllegalStateException(
                "Dataset is missing or empty."
            );
        }

        if (data.attribute(TARGET) == null) {
            throw new IllegalStateException(
                "Target attribute not found."
            );
        }

        data.setClassIndex(
            data.attribute(TARGET).index()
        );

        // ==========================================
        // STEP 2: TRAIN / TEST SPLIT (80/20)
        // ==========================================

        data.randomize(new Random(SEED));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);
        Instances testData = data.testCV(5, 0);

        System.out.println(
            "\n===== DATASET INFORMATION ====="
        );

        System.out.println(
            "Total instances: " + data.numInstances()
        );

        System.out.println(
            "Training instances: " + trainData.numInstances()
        );

        System.out.println(
            "Held-out test instances: " + testData.numInstances()
        );

        // ==========================================
        // STEP 3: INITIALIZE CLASSIFIERS
        // ==========================================

        J48Classifier j48 = new J48Classifier();

        OneRClassifier oneR = new OneRClassifier();

        RandomForestClassifier randomForest =
            new RandomForestClassifier();

        NaiveBayesClassifier naiveBayes =
            new NaiveBayesClassifier();

        SimpleLogisticClassifier simpleLogistic =
            new SimpleLogisticClassifier();

        // ==========================================
        // STEP 4: EVALUATE ALL CLASSIFIERS
        // ==========================================

        System.out.println(
            "\n===== BASELINE MODEL EVALUATION ====="
        );

        // Classifier 1: J48
        evaluate(
            "J48",
            j48.getModel(),
            trainData
        );

        // Classifier 2: OneR
        evaluate(
            "OneR",
            oneR.getModel(),
            trainData
        );

        // Classifier 3: Random Forest
        evaluate(
            "Random Forest",
            randomForest.getModel(),
            trainData
        );

        // Classifier 4: Naive Bayes
        evaluate(
            "Naive Bayes",
            naiveBayes.getClassifier(),
            trainData
        );

        // Classifier 5: Simple Logistic
        evaluate(
            "Simple Logistic",
            simpleLogistic.getClassifier(),
            trainData
        );

        // ==========================================
        // STEP 5: COMPLETION
        // ==========================================

        System.out.println(
            "\n================================="
        );

        System.out.println(
            "All 5 classifiers evaluated successfully!"
        );

        System.out.println(
            "Held-out test set remains untouched."
        );

        System.out.println(
            "================================="
        );
        
        
        // ==========================================
        // HOLD-OUT FUNCTION SMOKE TEST
        // Uses training data only.
        // Does NOT access the final 2,000 test rows.
        // ==========================================

        Instances smokeData = new Instances(trainData);

        smokeData.randomize(new Random(SEED));
        smokeData.stratify(5);

        Instances smokeTrain = smokeData.trainCV(5, 0);
        Instances smokeValidation = smokeData.testCV(5, 0);

        System.out.println(
            "\n===== HOLD-OUT FUNCTION TEST ====="
        );

        ModelEvaluator.TimedResult smokeResult =
            ModelEvaluator.evaluateHoldoutWithTiming(
            j48.getModel(),
            smokeTrain,
            smokeValidation
        );

        ModelEvaluator.printTimedResults(
            "J48 - Internal Validation",
            smokeResult,
            smokeValidation
        );

        System.out.println(
            "\nFinal 2,000-instance test set remains untouched."
        );

        System.out.println(
            "================================="
        );
    }

    // ==============================================
    // SHARED EVALUATION METHOD
    // ==============================================

    private static void evaluate(
            String modelName,
            Classifier classifier,
            Instances trainData) throws Exception {

        // Show training time
        ModelEvaluator.TimedResult result =
        ModelEvaluator.evaluateCVWithTiming(
            classifier,
            trainData
        );

        // Print evaluation metrics
        ModelEvaluator.printTimedResults(
            modelName,
            result,
            trainData
        );

        // // Perform 10-fold cross-validation
        // Evaluation result = ModelEvaluator.evaluateCV(
        //     classifier,
        //     trainData
        // );

        // // Print evaluation metrics
        // ModelEvaluator.printResults(
        //     modelName,
        //     result,
        //     trainData
        // );
    }
}

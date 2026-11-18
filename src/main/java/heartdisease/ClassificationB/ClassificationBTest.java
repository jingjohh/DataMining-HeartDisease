
package heartdisease.ClassificationB;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassificationBTest {

    public static void main(String[] args) throws Exception {

        // 1. Load the preprocessed dataset
        Instances data = DataSource.read(
            "data/processed/heart_disease.arff"
        );

        if (data == null || data.numInstances() == 0) {
            throw new IllegalStateException(
                "Dataset is missing or empty."
            );
        }

        data.setClassIndex(
            data.attribute("Heart Disease Status").index()
        );

        // 2. Prepare the same train/test data for both classifiers
        data.randomize(new Random(42));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);
        Instances testData = data.testCV(5, 0);

        Files.createDirectories(Paths.get("models"));

        // Use one test instance for prediction
        Instance patient = testData.instance(0);

        // ==========================================
        // 3. NAIVE BAYES
        // ==========================================
        System.out.println("\n===== NAIVE BAYES =====");

        NaiveBayesClassifier nb = new NaiveBayesClassifier();

        // Train
        nb.train(trainData);

        // Print configuration
        System.out.println("Naive Bayes configuration:");
        System.out.println(
            Arrays.toString(nb.getClassifier().getOptions())
        );

        // Predict
        double nbPrediction = nb.predict(patient);

        System.out.println(
            "Actual: " +
            patient.stringValue(patient.classIndex())
        );

        System.out.println(
            "Predicted: " +
            patient.classAttribute().value((int) nbPrediction)
        );

        // Save model
        nb.saveModel("models/NaiveBayes.model");

        // Load model and verify prediction
        NaiveBayesClassifier loadedNB =
            new NaiveBayesClassifier();

        loadedNB.loadModel("models/NaiveBayes.model");

        double nbLoadedPrediction =
            loadedNB.predict(patient);

        if (Double.isNaN(nbPrediction) ||
            Double.compare(nbPrediction, nbLoadedPrediction) != 0) {
            throw new IllegalStateException(
                "Naive Bayes save/load failed!"
            );
        }

        System.out.println("Naive Bayes Save/Load: PASS");

        // ==========================================
        // 4. SIMPLE LOGISTIC
        // ==========================================
        System.out.println("\n===== SIMPLE LOGISTIC =====");

        SimpleLogisticClassifier sl =
            new SimpleLogisticClassifier();

        // Train
        sl.train(trainData);

        // Print configuration
        System.out.println("SimpleLogistic configuration:");
        System.out.println(
            Arrays.toString(sl.getClassifier().getOptions())
        );

        // Predict
        double slPrediction = sl.predict(patient);

        System.out.println(
            "Actual: " +
            patient.stringValue(patient.classIndex())
        );

        System.out.println(
            "Predicted: " +
            patient.classAttribute().value((int) slPrediction)
        );

        // Save model
        sl.saveModel("models/SimpleLogistic.model");

        // Load model and verify prediction
        SimpleLogisticClassifier loadedSL =
            new SimpleLogisticClassifier();

        loadedSL.loadModel("models/SimpleLogistic.model");

        double slLoadedPrediction =
            loadedSL.predict(patient);

        if (Double.isNaN(slPrediction) ||
            Double.compare(slPrediction, slLoadedPrediction) != 0) {
            throw new IllegalStateException(
                "SimpleLogistic save/load failed!"
            );
        }

        System.out.println("SimpleLogistic Save/Load: PASS");

        System.out.println(
            "\nClassification B testing completed!"
        );
    }
}

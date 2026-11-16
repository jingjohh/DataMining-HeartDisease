
package heartdisease.ClassificationA;

import java.util.Random;
import java.nio.file.Files;
import java.nio.file.Paths;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassificationATest {

    public static void main(String[] args) throws Exception {

        // Load dataset
        Instances data = DataSource.read(
            "data/processed/heart_disease.arff"
        );

        // Set class attribute
        data.setClassIndex(
            data.attribute("Heart Disease Status").index()
        );

        // Split dataset: 80% train, 20% test
        data.randomize(new Random(42));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);
        Instances testData = data.testCV(5, 0);

        System.out.println("Training size: "
            + trainData.numInstances());
        System.out.println("Testing size: "
            + testData.numInstances());

        // Create folder for models
        Files.createDirectories(Paths.get("models"));

        // Select one instance from test dataset
        Instance patient = testData.instance(0);


        // J48

        System.out.println("\n===== J48 =====");

        J48Classifier j48 = new J48Classifier();

        // Train
        j48.train(trainData);
        System.out.println("J48 training completed");

        // Predict
        double j48Prediction = j48.predict(patient);
        System.out.println("J48 prediction completed");

        // Save
        j48.saveModel("models/J48.model");
        System.out.println("J48 model saved");

        // Load
        J48Classifier loadedJ48 = new J48Classifier();
        loadedJ48.loadModel("models/J48.model");
        System.out.println("J48 model loaded");

        // Predict using loaded model
        double j48LoadedPrediction = loadedJ48.predict(patient);

        printResult("J48", patient,
            j48Prediction, j48LoadedPrediction);


        // OneR 

        System.out.println("\n===== OneR =====");

        OneRClassifier oneR = new OneRClassifier();

        // Train
        oneR.train(trainData);
        System.out.println("OneR training completed");

        // Predict
        double oneRPrediction = oneR.predict(patient);
        System.out.println("OneR prediction completed");

        // Save
        oneR.saveModel("models/OneR.model");
        System.out.println("OneR model saved");

        // Load
        OneRClassifier loadedOneR = new OneRClassifier();
        loadedOneR.loadModel("models/OneR.model");
        System.out.println("OneR model loaded");

        // Predict using loaded model
        double oneRLoadedPrediction = loadedOneR.predict(patient);

        printResult("OneR", patient,
            oneRPrediction, oneRLoadedPrediction);


        // Random Forest 

        System.out.println("\n===== Random Forest =====");

        RandomForestClassifier rf =
            new RandomForestClassifier();

        // Train
        rf.train(trainData);
        System.out.println("Random Forest training completed");

        // Predict
        double rfPrediction = rf.predict(patient);
        System.out.println("Random Forest prediction completed");

        // Save
        rf.saveModel("models/RandomForest.model");
        System.out.println("Random Forest model saved");

        // Load
        RandomForestClassifier loadedRF =
            new RandomForestClassifier();

        loadedRF.loadModel("models/RandomForest.model");
        System.out.println("Random Forest model loaded");

        // Predict using loaded model
        double rfLoadedPrediction = loadedRF.predict(patient);

        printResult("Random Forest", patient,
            rfPrediction, rfLoadedPrediction);

        System.out.println(
            "\nAll three classifiers tested successfully!"
        );
    }

    // Check prediction before and after loading
    public static void printResult(
            String name,
            Instance patient,
            double before,
            double after) {

        if (Double.isNaN(before) ||
                Double.compare(before, after) != 0) {
            throw new IllegalStateException(
                name + " save/load test failed!"
            );
        }

        String actual = patient.stringValue(
            patient.classIndex()
        );

        String predicted = patient.classAttribute()
            .value((int) before);

        System.out.println("Actual: " + actual);
        System.out.println("Predicted: " + predicted);
        System.out.println(name + " Save/Load: PASS");
    }
}

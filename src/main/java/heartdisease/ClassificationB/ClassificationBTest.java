
package heartdisease.ClassificationB;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ClassificationBTest {

    public static void main(String[] args) throws Exception {

        // 1. Load dataset
        Instances data = DataSource.read(
            "data/processed/heart_disease.arff"
        );

        data.setClassIndex(
            data.attribute("Heart Disease Status").index()
        );

        System.out.println("Total instances: " + data.numInstances());
        System.out.println("Total attributes: " + data.numAttributes());

        // check lại data
        System.out.println("\n=== CLASS DISTRIBUTION ===");

for (int i = 0; i < data.numClasses(); i++) {
    String className = data.classAttribute().value(i);
    int count = 0;

    for (int j = 0; j < data.numInstances(); j++) {
        if (data.instance(j).classValue() == i) {
            count++;
        }
    }

    System.out.println(className + ": " + count);
}

System.out.println("\n=== ATTRIBUTE INFORMATION ===");

for (int i = 0; i < data.numAttributes(); i++) {
    System.out.println(
        data.attribute(i).name()
        + " | Type: "
        + data.attribute(i).type()
        + " | Missing: "
        + data.attributeStats(i).missingCount
    );
}

        // 2. Split data (80% training, 20% testing)
        data.randomize(new Random(42));
        data.stratify(5);

        Instances trainData = data.trainCV(5, 0);
        Instances testData = data.testCV(5, 0);

        System.out.println("Training size: " + trainData.numInstances());
        System.out.println("Testing size: " + testData.numInstances());

        Files.createDirectories(Paths.get("models"));

        Instance patient = testData.instance(0);

        // 3. Naive Bayes
        System.out.println("\n===== NAIVE BAYES =====");

        NaiveBayesClassifier nb = new NaiveBayesClassifier();
        nb.train(trainData);

        double nbPrediction = nb.predict(patient);

        System.out.println("Actual: " + patient.stringValue(
            patient.classIndex()
        ));

        System.out.println("Predicted: " +
            patient.classAttribute().value((int) nbPrediction)
        );

        Evaluation nbEval = new Evaluation(trainData);
        nbEval.evaluateModel(nb.getClassifier(), testData);

        System.out.println(nbEval.toSummaryString());
        System.out.println(nbEval.toMatrixString());

        System.out.println(nbEval.toClassDetailsString());

        // Save and reload
        nb.saveModel("models/NaiveBayes.model");

        NaiveBayesClassifier loadedNB = new NaiveBayesClassifier();
        loadedNB.loadModel("models/NaiveBayes.model");

        double nbLoadedPrediction = loadedNB.predict(patient);

        if (Double.isNaN(nbPrediction) ||
            Double.compare(nbPrediction, nbLoadedPrediction) != 0) {
            throw new IllegalStateException("Naive Bayes save/load failed!");
        }

        System.out.println("Naive Bayes Save/Load: PASS");

        // 4. SimpleLogistic
        System.out.println("\n===== SIMPLE LOGISTIC =====");

        SimpleLogisticClassifier sl = new SimpleLogisticClassifier();
        sl.train(trainData);

        double slPrediction = sl.predict(patient);

        System.out.println("Actual: " + patient.stringValue(
            patient.classIndex()
        ));

        System.out.println("Predicted: " +
            patient.classAttribute().value((int) slPrediction)
        );

        Evaluation slEval = new Evaluation(trainData);
        slEval.evaluateModel(sl.getClassifier(), testData);

        System.out.println(slEval.toSummaryString());
        System.out.println(slEval.toMatrixString());

        System.out.println(slEval.toClassDetailsString());

        // Save and reload
        sl.saveModel("models/SimpleLogistic.model");

        SimpleLogisticClassifier loadedSL =
            new SimpleLogisticClassifier();

        loadedSL.loadModel("models/SimpleLogistic.model");

        double slLoadedPrediction = loadedSL.predict(patient);

        if (Double.isNaN(slPrediction) ||
            Double.compare(slPrediction, slLoadedPrediction) != 0) {
            throw new IllegalStateException("SimpleLogistic save/load failed!");
        }

        System.out.println("SimpleLogistic Save/Load: PASS");

        System.out.println("\nClassification B testing completed!");
    }
}



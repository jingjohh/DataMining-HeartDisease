
package heartdisease.ClassificationB;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class NaiveBayesClassifier {

    private NaiveBayes model;

    // Initialize Naive Bayes classifier
    public NaiveBayesClassifier() {
        model = new NaiveBayes();
    }

    // Train the classifier
    public void train(Instances trainingData) throws Exception {
        if (trainingData == null || trainingData.numInstances() == 0) {
            throw new IllegalArgumentException("Training data is empty.");
        }

        if (trainingData.classIndex() == -1) {
            trainingData.setClassIndex(trainingData.numAttributes() - 1);
        }

        model.buildClassifier(trainingData);
        System.out.println("Naive Bayes training completed.");
    }

    // Predict the class label
    public double predict(Instance instance) throws Exception {
        return model.classifyInstance(instance);
    }

    // Predict class probabilities
    public double[] predictProbabilities(Instance instance)
            throws Exception {
        return model.distributionForInstance(instance);
    }

    // Save trained classifier
    public void saveModel(String filePath) throws Exception {
        SerializationHelper.write(filePath, model);
        System.out.println("Model saved to: " + filePath);
    }

    // Load saved classifier
    public void loadModel(String filePath) throws Exception {
        model = (NaiveBayes) SerializationHelper.read(filePath);
        System.out.println("Model loaded from: " + filePath);
    }

    // Return classifier for evaluation
    public NaiveBayes getClassifier() {
        return model;
    }
}

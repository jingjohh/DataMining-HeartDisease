package heartdisease.ClassificationA;

import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class J48Classifier {

    private J48 model;

    // Initialize J48 model
    public J48Classifier() {
        model = new J48();
    }

    // Train model
    public void train(Instances trainData) throws Exception {
        model.buildClassifier(trainData);
    }

    // Predict 
    public double predict(Instance instance) throws Exception {
        return model.classifyInstance(instance);
    }

    // Save model
    public void saveModel(String path) throws Exception {
        SerializationHelper.write(path, model);
    }

    // Load model
    public void loadModel(String path) throws Exception {
        model = (J48) SerializationHelper.read(path);
    }

    // Return model for evaluation framework
    public J48 getModel() {
        return model;
    }
}

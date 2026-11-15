package heartdisease.ClassificationA;

import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class RandomForestClassifier {

    private RandomForest model;

    public RandomForestClassifier() {
        model = new RandomForest();
        model.setNumIterations(100);
        model.setSeed(42);
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
        model = (RandomForest)
            SerializationHelper.read(path);
    }

    // Return model for evaluation
    public RandomForest getModel() {
        return model;
    }
}


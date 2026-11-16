
package heartdisease.ClassificationA;

import weka.classifiers.rules.OneR;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class OneRClassifier {

    private OneR model;

    // Initialize OneR model
    public OneRClassifier() {
        model = new OneR();
        model.setMinBucketSize(6);
    }

    // Train
    public void train(Instances trainData) throws Exception {
        model.buildClassifier(trainData);
    }

    // Predict
    public double predict(Instance instance) throws Exception {
        return model.classifyInstance(instance);
    }

    // Save
    public void saveModel(String path) throws Exception {
        SerializationHelper.write(path, model);
    }

    // Load
    public void loadModel(String path) throws Exception {
        model = (OneR) SerializationHelper.read(path);
    }

    // Return model for evaluation
    public OneR getModel() {
        return model;
    }
}

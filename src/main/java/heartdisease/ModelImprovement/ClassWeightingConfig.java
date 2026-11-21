package heartdisease.ModelImprovement;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.WeightedInstancesHandlerWrapper;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public final class ClassWeightingConfig {

    private static final int[] YES_CLASS_WEIGHTS = {
            2, 4, 8
    };

    private static final int RANDOM_SEED = 42;

    private ClassWeightingConfig() {
    }

    public static int[] getYesClassWeights() {
        return YES_CLASS_WEIGHTS.clone();
    }

    public static Classifier createWeightedClassifier(
            Classifier baseClassifier,
            int noIndex,
            int yesIndex,
            int yesWeight
    ) {

        if (baseClassifier == null) {
            throw new IllegalArgumentException(
                    "Base classifier cannot be null."
            );
        }

        if (noIndex < 0 || noIndex > 1
                || yesIndex < 0 || yesIndex > 1
                || noIndex == yesIndex) {

            throw new IllegalArgumentException(
                    "Expected two different class indices."
            );
        }

        if (yesWeight < 1) {
            throw new IllegalArgumentException(
                    "Yes class weight must be at least 1."
            );
        }

        return new WeightedTrainingClassifier(
                baseClassifier,
                yesIndex,
                yesWeight
        );
    }

    // ============================================
    // APPLY WEIGHTS INSIDE EACH TRAINING FOLD
    // ============================================

    private static final class WeightedTrainingClassifier
            extends AbstractClassifier {

        private static final long serialVersionUID = 1L;

        private final Classifier baseClassifier;
        private final int yesIndex;
        private final int yesWeight;

        private Classifier trainedClassifier;

        private WeightedTrainingClassifier(
                Classifier baseClassifier,
                int yesIndex,
                int yesWeight
        ) {
            this.baseClassifier = baseClassifier;
            this.yesIndex = yesIndex;
            this.yesWeight = yesWeight;
        }

        @Override
        public Capabilities getCapabilities() {
            return baseClassifier.getCapabilities();
        }

        @Override
        public void buildClassifier(Instances trainData)
                throws Exception {

            if (trainData.classIndex() < 0) {
                throw new IllegalStateException(
                        "Class index is not set."
                );
            }

            if (yesIndex >= trainData.numClasses()) {
                throw new IllegalStateException(
                        "Invalid positive class index."
                );
            }

            // Copy the fold's training data.
            // Do not modify the original dataset.
            Instances weightedData =
                    new Instances(trainData);

            // Increase the weight of actual Yes instances.
            for (int i = 0;
                    i < weightedData.numInstances();
                    i++) {

                Instance instance =
                        weightedData.instance(i);

                if (!instance.classIsMissing()
                        && (int) instance.classValue()
                        == yesIndex) {

                    instance.setWeight(
                            instance.weight() * yesWeight
                    );
                }
            }

            // Create an independent copy of the base model.
            Classifier modelCopy =
                    AbstractClassifier.makeCopy(
                            baseClassifier
                    );

            // If a base classifier supports instance weights,
            // Weka uses them directly.
            //
            // Otherwise, Weka performs weighted resampling.
            WeightedInstancesHandlerWrapper wrapper =
                    new WeightedInstancesHandlerWrapper();

            wrapper.setClassifier(modelCopy);
            wrapper.setSeed(RANDOM_SEED);

            // Train using only weighted training-fold data.
            wrapper.buildClassifier(weightedData);

            trainedClassifier = wrapper;
        }

        @Override
        public double classifyInstance(Instance instance)
                throws Exception {

            if (trainedClassifier == null) {
                throw new IllegalStateException(
                        "Model has not been trained."
                );
            }

            return trainedClassifier.classifyInstance(instance);
        }

        @Override
        public double[] distributionForInstance(
                Instance instance
        ) throws Exception {

            if (trainedClassifier == null) {
                throw new IllegalStateException(
                        "Model has not been trained."
                );
            }

            return trainedClassifier.distributionForInstance(
                    instance
            );
        }
    }
}
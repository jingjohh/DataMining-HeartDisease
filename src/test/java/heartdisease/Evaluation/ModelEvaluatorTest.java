
package heartdisease.Evaluation;

import static org.junit.jupiter.api.Assertions.*;

import java.io.StringReader;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ModelEvaluatorTest {

    // ==========================================
    // TEST DATA
    // ==========================================

    private Instances parseArff(String arff) throws Exception {
        Instances data = new Instances(new StringReader(arff));
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    private Instances createDataset() throws Exception {

        StringBuilder arff = new StringBuilder("""
            @relation toy
            @attribute sample_id numeric
            @attribute signal numeric
            @attribute 'Heart Disease Status' {No,Yes}
            @data
            """);

        for (int i = 0; i < 40; i++) {
            int signal = i % 2;
            String label = signal == 0 ? "No" : "Yes";

            arff.append(i + 1)
                .append(",")
                .append(signal)
                .append(",")
                .append(label)
                .append("\n");
        }

        return parseArff(arff.toString());
    }

    private double countPredictions(Evaluation evaluation)
            throws Exception {

        double total = 0;

        for (double[] row : evaluation.confusionMatrix()) {
            for (double count : row) {
                total += count;
            }
        }

        return total;
    }

    // ==========================================
    // TEST 1: TEN INDEPENDENT FOLDS
    // ==========================================

    @Test
    void crossValidationUsesTenIndependentFolds()
            throws Exception {

        Instances data = createDataset();

        FoldProbeClassifier.buildCount.set(0);

        ModelEvaluator.TimedResult result =
                ModelEvaluator.evaluateCVWithTiming(
                        new FoldProbeClassifier(), data
                );

        // Exactly 10 model training operations
        assertEquals(
                10, FoldProbeClassifier.buildCount.get()
        );

        // Every instance must be predicted exactly once
        assertEquals(40, result.getPredictedInstances());

        assertEquals(
                40.0,
                countPredictions(result.getEvaluation()),
                1e-9
        );

        // CV must not return a single final trained model
        assertNull(result.getTrainedModel());
    }

    // ==========================================
    // TEST 2: REPRODUCIBLE CV RESULTS
    // ==========================================

    @Test
    void crossValidationIsReproducible() throws Exception {

        Instances data = createDataset();

        Evaluation first = ModelEvaluator.evaluateCV(
                new J48(), data
        );

        Evaluation second = ModelEvaluator.evaluateCV(
                new J48(), data
        );

        assertEquals(
                first.pctCorrect(),
                second.pctCorrect(),
                1e-9
        );

        double[][] a = first.confusionMatrix();
        double[][] b = second.confusionMatrix();

        for (int i = 0; i < a.length; i++) {
            assertArrayEquals(a[i], b[i], 1e-9);
        }
    }

    // ==========================================
    // TEST 3: HOLD-OUT EVALUATION
    // ==========================================

    @Test
    void holdoutEvaluationReturnsTrainedModel()
            throws Exception {

        Instances data = createDataset();

        Instances train = new Instances(data, 0, 30);
        Instances test = new Instances(data, 30, 10);

        ModelEvaluator.TimedResult result =
                ModelEvaluator.evaluateHoldoutWithTiming(
                        new J48(), train, test
                );

        assertNotNull(result.getTrainedModel());

        assertEquals(10, result.getPredictedInstances());

        assertEquals(
                10.0,
                countPredictions(result.getEvaluation()),
                1e-9
        );

        assertTrue(result.getTrainingMs() >= 0);
        assertTrue(result.getPredictionMs() >= 0);
    }

    // ==========================================
    // TEST 4: INCOMPATIBLE SCHEMA
    // ==========================================

    @Test
    void holdoutRejectsIncompatibleSchema()
            throws Exception {

        Instances train = createDataset();
        Instances test = new Instances(train);

        test.renameAttribute(0, "different_id");

        assertThrows(
                IllegalArgumentException.class,
                () -> ModelEvaluator.evaluateHoldoutWithTiming(
                        new J48(), train, test
                )
        );
    }

    // ==========================================
    // TEST 5: MODEL SERIALIZATION
    // ==========================================

    @Test
    void savedModelPreservesPredictions(
            @TempDir Path tempDir) throws Exception {

        Instances data = createDataset();

        Instances train = new Instances(data, 0, 30);
        Instances test = new Instances(data, 30, 10);

        ModelEvaluator.TimedResult result =
                ModelEvaluator.evaluateHoldoutWithTiming(
                        new J48(), train, test
                );

        Classifier original = result.getTrainedModel();

        Path modelPath = tempDir.resolve("J48.model");

        SerializationHelper.write(
                modelPath.toString(), original
        );

        Classifier loaded = (Classifier)
                SerializationHelper.read(
                        modelPath.toString()
                );

        for (int i = 0; i < test.numInstances(); i++) {

            Instance sample = test.instance(i);

            assertArrayEquals(
                    original.distributionForInstance(sample),
                    loaded.distributionForInstance(sample),
                    1e-9
            );
        }
    }

    // ==========================================
    // TEST 6: TRAIN-ONLY IMPUTATION
    // ==========================================

    @Test
    void missingValueImputationUsesTrainingStatistics()
            throws Exception {

        Instances train = parseArff("""
            @relation imputation
            @attribute x numeric
            @attribute 'Heart Disease Status' {No,Yes}
            @data
            2,No
            4,No
            6,Yes
            8,Yes
            """);

        Instances validation = parseArff("""
            @relation imputation
            @attribute x numeric
            @attribute 'Heart Disease Status' {No,Yes}
            @data
            ?,No
            10000,Yes
            """);

        FilteredClassifier pipeline =
                new FilteredClassifier();

        pipeline.setFilter(new ReplaceMissingValues());
        pipeline.setClassifier(new ThresholdClassifier());

        ModelEvaluator.TimedResult result =
                ModelEvaluator.evaluateHoldoutWithTiming(
                        pipeline, train, validation
                );

        Classifier fitted = result.getTrainedModel();

        // Training mean = (2+4+6+8)/4 = 5.
        // Missing validation value should become 5,
        // not be influenced by validation value 10000.
        assertEquals(
                0.0,
                fitted.classifyInstance(validation.instance(0)),
                1e-9
        );

        assertEquals(
                1.0,
                fitted.classifyInstance(validation.instance(1)),
                1e-9
        );
    }

    // ==========================================
    // TEST 7: INVALID CV INPUT
    // ==========================================

    @Test
    void crossValidationRejectsTooFewInstances()
            throws Exception {

        Instances data = createDataset();
        Instances small = new Instances(data, 0, 9);

        assertThrows(
                IllegalArgumentException.class,
                () -> ModelEvaluator.evaluateCV(
                        new J48(), small
                )
        );
    }

    // ==========================================
    // TEST HELPER: DETECT FOLD LEAKAGE
    // ==========================================

    public static class FoldProbeClassifier
            extends AbstractClassifier {

        private static final long serialVersionUID = 1L;

        static final AtomicInteger buildCount =
                new AtomicInteger();

        private Set<Double> trainingIds;

        @Override
        public void buildClassifier(Instances trainData) {

            buildCount.incrementAndGet();

            if (trainData.numInstances() != 36) {
                throw new IllegalStateException(
                        "Unexpected training fold size."
                );
            }

            trainingIds = new HashSet<>();

            for (Instance instance : trainData) {
                trainingIds.add(instance.value(0));
            }
        }

        @Override
        public double[] distributionForInstance(
                Instance instance) {

            if (trainingIds == null) {
                throw new IllegalStateException(
                        "Model was not trained."
                );
            }

            if (trainingIds.contains(instance.value(0))) {
                throw new IllegalStateException(
                        "Data leakage: test ID found in training."
                );
            }

            return instance.value(1) >= 0.5
                    ? new double[]{0.0, 1.0}
                    : new double[]{1.0, 0.0};
        }
    }

    // ==========================================
    // TEST HELPER: FIXED THRESHOLD CLASSIFIER
    // ==========================================

    public static class ThresholdClassifier
            extends AbstractClassifier {

        private static final long serialVersionUID = 1L;

        @Override
        public Capabilities getCapabilities() {

            Capabilities capabilities =
                    new Capabilities(this);

            capabilities.disableAll();

            capabilities.enable(
                    Capabilities.Capability.NUMERIC_ATTRIBUTES
            );

            capabilities.enable(
                    Capabilities.Capability.NOMINAL_CLASS
            );

            capabilities.enable(
                    Capabilities.Capability.MISSING_VALUES
            );

            capabilities.setMinimumNumberInstances(1);

            return capabilities;
        }

        @Override
        public void buildClassifier(Instances trainData) {
            // No parameters to learn for this test.
        }

        @Override
        public double[] distributionForInstance(
                Instance instance) {

            return instance.value(0) > 50
                    ? new double[]{0.0, 1.0}
                    : new double[]{1.0, 0.0};
        }
    }
}

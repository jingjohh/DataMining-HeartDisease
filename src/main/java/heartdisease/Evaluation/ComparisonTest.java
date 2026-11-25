
package heartdisease.Evaluation;

// Classification A
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import heartdisease.ClassificationA.J48Classifier;
import heartdisease.ClassificationA.OneRClassifier;
import heartdisease.ClassificationA.RandomForestClassifier;
import heartdisease.ClassificationB.NaiveBayesClassifier;
import heartdisease.ClassificationB.SimpleLogisticClassifier;
import heartdisease.ModelImprovement.ClassWeightingConfig;
import heartdisease.ModelImprovement.FeatureSelectionPipeline;
import heartdisease.ModelImprovement.ImprovementResult;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ComparisonTest {

    private static final String DATA_PATH =
            "data/processed/heart_disease.arff";

    private static final String TARGET =
            "Heart Disease Status";

    private static final int SEED = 42;

    public static void main(String[] args) throws Exception {

        // ==========================================
        // STEP 1: LOAD AND SPLIT DATA
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

        int noIndex =
                data.classAttribute().indexOfValue("No");

        int yesIndex =
                data.classAttribute().indexOfValue("Yes");

        if (noIndex < 0 || yesIndex < 0) {
            throw new IllegalStateException(
                    "Expected target values: No and Yes."
            );
        }

        data.randomize(new Random(SEED));
        data.stratify(5);

        // Use only the original 80% training partition.
        Instances trainData = data.trainCV(5, 0);

        System.out.println(
                "Training instances: "
                + trainData.numInstances()
        );

        // ==========================================
        // STEP 2: INITIALIZE BASELINE CLASSIFIERS
        // ==========================================

        String[] names = {
                "J48",
                "OneR",
                "Random Forest",
                "Naive Bayes",
                "SimpleLogistic"
        };

        Classifier[] classifiers = {
                new J48Classifier().getModel(),
                new OneRClassifier().getModel(),
                new RandomForestClassifier().getModel(),
                new NaiveBayesClassifier().getClassifier(),
                new SimpleLogisticClassifier().getClassifier()
        };

        List<ModelComparator.Comparison> comparisons =
                new ArrayList<>();

        // ==========================================
        // STEP 3: EVALUATE & COMPARE
        // ==========================================

        for (int i = 0; i < classifiers.length; i++) {

            String name = names[i];
            Classifier base = classifiers[i];

            System.out.println(
                    "\nEvaluating: " + name
            );

            // A. BASELINE
            ImprovementResult baseline = runCV(
                    name,
                    "Baseline - All Features",
                    base,
                    trainData,
                    yesIndex
            );

            // B. IMPROVED: CFS + BESTFIRST
            Classifier cfsModel =
                    FeatureSelectionPipeline.createCFSModel(
                            AbstractClassifier.makeCopy(base)
                    );

            ImprovementResult cfsResult = runCV(
                    name,
                    "CFS + BestFirst",
                    cfsModel,
                    trainData,
                    yesIndex
            );

            comparisons.add(
                    ModelComparator.compare(
                            baseline, cfsResult
                    )
            );

            // C. IMPROVED: CLASS WEIGHTING
            Classifier weightedModel =
                    ClassWeightingConfig.createWeightedClassifier(
                            AbstractClassifier.makeCopy(base),
                            noIndex,
                            yesIndex,
                            4
                    );

            ImprovementResult weightedResult = runCV(
                    name,
                    "Class Weighting YesWeight=4",
                    weightedModel,
                    trainData,
                    yesIndex
            );

            comparisons.add(
                    ModelComparator.compare(
                            baseline, weightedResult
                    )
            );
        }

        // ==========================================
        // STEP 4: EXPORT COMPARISON RESULTS
        // ==========================================

        ModelComparator.saveCsv(
                comparisons,
                "results/step4_model_comparison.csv"
        );

        System.out.println(
                "\nComparison completed successfully!"
        );

        System.out.println(
                "Final test set has not been evaluated."
        );
    }

    // ==========================================
    // SHARED CV HELPER
    // ==========================================

    private static ImprovementResult runCV(
            String modelName,
            String methodName,
            Classifier classifier,
            Instances trainData,
            int positiveClass) throws Exception {

        ModelEvaluator.TimedResult timed =
                ModelEvaluator.evaluateCVWithTiming(
                        classifier,
                        trainData
                );

        return ImprovementResult.fromEvaluation(
                modelName,
                methodName,
                timed.getEvaluation(),
                positiveClass
        );
    }
}

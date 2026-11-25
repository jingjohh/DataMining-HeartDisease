package heartdisease.ModelImprovement;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;

public final class FeatureSelectionConfig {

    // ============================================
    // FEATURE SELECTION CONFIGURATIONS
    // ============================================

    // Number of features for Information Gain experiments
    private static final int[] TOP_K_VALUES = {5, 10, 15};

    // Prevent object creation
    private FeatureSelectionConfig() {
    }


    // ============================================
    // METHOD 1: CFS + BESTFIRST
    // ============================================

    // Create CFS feature evaluator
    public static CfsSubsetEval createEvaluator() {

        return new CfsSubsetEval();
    }

    // Create BestFirst search method
    public static BestFirst createSearchMethod() {

        return new BestFirst();
    }


    // ============================================
    // METHOD 2: INFORMATION GAIN + RANKER
    // ============================================

    // Create Information Gain evaluator
    public static InfoGainAttributeEval createInfoGainEvaluator() {

        return new InfoGainAttributeEval();
    }

    // Create Ranker with a specified number of features
    public static Ranker createRanker(int topK) {

        // Validate Top K
        if (topK <= 0) {
            throw new IllegalArgumentException(
                    "Top K must be greater than zero."
            );
        }

        // Create Ranker
        Ranker ranker = new Ranker();

        // Select Top K features
        ranker.setNumToSelect(topK);

        return ranker;
    }


    // ============================================
    // GET EXPERIMENT CONFIGURATIONS
    // ============================================

    public static int[] getTopKValues() {

        // Return a copy to protect the original array
        return TOP_K_VALUES.clone();
    }
}

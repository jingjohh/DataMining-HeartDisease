
package heartdisease.ModelImprovement;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;

import weka.filters.supervised.attribute.AttributeSelection;

public class FeatureSelectionPipeline {

    // ============================================
    // METHOD 1: CFS + BESTFIRST
    // ============================================

    public static FilteredClassifier createCFSModel(
            Classifier classifier) throws Exception {

        // Create feature selection filter
        AttributeSelection selector =
                new AttributeSelection();

        // Configure CFS evaluator
        selector.setEvaluator(
                FeatureSelectionConfig.createEvaluator()
        );

        // Configure BestFirst search method
        selector.setSearch(
                FeatureSelectionConfig.createSearchMethod()
        );

        // Combine feature selection and classifier
        return createFilteredModel(classifier, selector);
    }


    // ============================================
    // METHOD 2: INFORMATION GAIN + RANKER
    // ============================================

    public static FilteredClassifier createInfoGainModel(
            Classifier classifier,
            int topK) throws Exception {

        // Create feature selection filter
        AttributeSelection selector =
                new AttributeSelection();

        // Configure Information Gain evaluator
        selector.setEvaluator(
                FeatureSelectionConfig.createInfoGainEvaluator()
        );

        // Configure Ranker
        selector.setSearch(
                FeatureSelectionConfig.createRanker(topK)
        );

        // Combine feature selection and classifier
        return createFilteredModel(classifier, selector);
    }


    // ============================================
    // SHARED PIPELINE CREATION
    // ============================================

    private static FilteredClassifier createFilteredModel(
            Classifier classifier,
            AttributeSelection selector) {

        if (classifier == null) {
            throw new IllegalArgumentException(
                    "Classifier cannot be null."
            );
        }

        // Create filtered classifier
        FilteredClassifier model =
                new FilteredClassifier();

        // Set feature selection filter
        model.setFilter(selector);

        // Set classification algorithm
        model.setClassifier(classifier);

        return model;
    }
}

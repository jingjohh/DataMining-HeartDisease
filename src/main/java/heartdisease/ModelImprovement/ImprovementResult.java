
package heartdisease.ModelImprovement;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import weka.classifiers.Evaluation;

public class ImprovementResult {

    // ============================================
    // MODEL INFORMATION
    // ============================================

    private String modelName;
    private String methodName;

    // -1 means feature count is not available
    private int featureCount = -1;


    // ============================================
    // EXPERIMENT RESULTS
    // ============================================

    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private double rocAuc;

    private double[][] confusionMatrix;

    private List<String> selectedFeatures = new ArrayList<>();


    // ============================================
    // CONSTRUCTOR
    // ============================================

    public ImprovementResult(
            String modelName,
            String methodName) {

        this.modelName = modelName;
        this.methodName = methodName;
    }


    // ============================================
    // CREATE RESULT FROM WEKA EVALUATION
    // ============================================

    public static ImprovementResult fromEvaluation(
            String modelName,
            String methodName,
            Evaluation evaluation,
            int positiveClass) {

        // Create result object
        ImprovementResult result =
                new ImprovementResult(modelName, methodName);

        // Accuracy is stored as a percentage
        result.setAccuracy(
                evaluation.pctCorrect()
        );

        // Precision of positive class
        result.setPrecision(
                evaluation.precision(positiveClass)
        );

        // Recall of positive class
        result.setRecall(
                evaluation.recall(positiveClass)
        );

        // F1-score of positive class
        result.setF1Score(
                evaluation.fMeasure(positiveClass)
        );

        // ROC-AUC of positive class
        result.setRocAuc(
                evaluation.areaUnderROC(positiveClass)
        );

        // Store confusion matrix
        result.setConfusionMatrix(
                evaluation.confusionMatrix()
        );

        return result;
    }


    // ============================================
    // CSV EXPORT SUPPORT
    // ============================================

    public static String csvHeader() {

        return "Model,Method,FeatureCount,AccuracyPercent,"
                + "PrecisionYes,RecallYes,F1Yes,RocAucYes,"
                + "SelectedFeatures";
    }


    // Convert experiment result to one CSV row
    public String toCsvRow() {

        String features =
                String.join("|", selectedFeatures);

        return String.join(",",
                csvEscape(modelName),
                csvEscape(methodName),
                featureCount < 0
                        ? ""
                        : Integer.toString(featureCount),
                csvNumber(accuracy),
                csvNumber(precision),
                csvNumber(recall),
                csvNumber(f1Score),
                csvNumber(rocAuc),
                csvEscape(features)
        );
    }


    // Handle valid and undefined metric values
    private static String csvNumber(double value) {

        if (!Double.isFinite(value)) {
            return "N/A";
        }

        return String.format(
                Locale.US,
                "%.4f",
                value
        );
    }


    // Escape text for CSV format
    private static String csvEscape(String value) {

        if (value == null) {
            return "";
        }

        return "\""
                + value.replace("\"", "\"\"")
                + "\"";
    }


    // ============================================
    // GETTERS AND SETTERS
    // ============================================

    public String getModelName() {
        return modelName;
    }

    public String getMethodName() {
        return methodName;
    }

    public int getFeatureCount() {
        return featureCount;
    }

    public void setFeatureCount(int featureCount) {
        this.featureCount = featureCount;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public double getPrecision() {
        return precision;
    }

    public void setPrecision(double precision) {
        this.precision = precision;
    }

    public double getRecall() {
        return recall;
    }

    public void setRecall(double recall) {
        this.recall = recall;
    }

    public double getF1Score() {
        return f1Score;
    }

    public void setF1Score(double f1Score) {
        this.f1Score = f1Score;
    }

    public double getRocAuc() {
        return rocAuc;
    }

    public void setRocAuc(double rocAuc) {
        this.rocAuc = rocAuc;
    }

    public double[][] getConfusionMatrix() {
        return confusionMatrix;
    }

    public void setConfusionMatrix(
            double[][] confusionMatrix) {

        this.confusionMatrix = confusionMatrix;
    }

    public List<String> getSelectedFeatures() {
        return selectedFeatures;
    }

    public void setSelectedFeatures(
            List<String> selectedFeatures) {

        this.selectedFeatures =
                selectedFeatures == null
                        ? new ArrayList<>()
                        : new ArrayList<>(selectedFeatures);
    }
}

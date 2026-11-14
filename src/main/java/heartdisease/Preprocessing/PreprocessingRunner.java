package heartdisease.Preprocessing;

import weka.core.Instances;
import weka.core.converters.CSVSaver;

import java.io.File;


public class PreprocessingRunner {

    private static final String INPUT_PATH = "data/raw/heart_disease.csv";

    private static final String CLEAN_CSV_PATH = "data/processed/heart_disease_clean.csv";

    private static final String ARFF_PATH = "data/processed/heart_disease.arff";

    public static void main(String[] args) {

        try {
            System.out.println("\n--- HEART DISEASE DATA PROCESSING ---");

            //LOAD DATASET
            System.out.println("\n--- DATA LOADING ---");

            Instances dataset = DataLoader.loadDataset(INPUT_PATH);

            if (dataset == null || dataset.numInstances() == 0) {
                System.out.println("Dataset is empty. Processing stopped.");
                return;
            }

            DataLoader.printDatasetOverview(dataset);
            DataLoader.printAttributeInfo(dataset);

            // EXPLORATORY DATA ANALYSIS
            System.out.println("\n --- EXPLORATORY DATA ANALYSIS ---");

            EDA.printMissingValues(dataset);
            EDA.printDuplicateAnalysis(dataset);
            EDA.printNumericalAnalysis(dataset);
            EDA.printCategoricalAnalysis(dataset);
            EDA.printTargetAnalysis(dataset);
            EDA.printOutlierAnalysis(dataset);
            EDA.printCorrelationAnalysis(dataset);
            EDA.printNumericalTargetRelationship(dataset);
            EDA.printCategoricalTargetRelationship(dataset);

            // PREPARE CLEAN SHARED DATASET
            System.out.println("\n--- DATA PREPROCESSING ---");

            Instances cleanDataset = DataPreprocessing.prepareSharedDataset(dataset);

            // EXPORT CLEAN DATASET AS CSV
            System.out.println("\n--- EXPORT CLEAN CSV ---");

            saveAsCsv(cleanDataset, CLEAN_CSV_PATH);

            // EXPORT DATASET AS ARFF
            System.out.println("\n--- EXPORT ARFF ---");

            DataPreprocessing.saveAsArff(cleanDataset, ARFF_PATH);

            //SUMMARY
            System.out.println("\n --- DATA PROCESSING COMPLETED ---");

            System.out.println("Original records: " + dataset.numInstances());
            System.out.println("Clean records: " + cleanDataset.numInstances());
            System.out.println("Attributes: " + cleanDataset.numAttributes());
            System.out.println("Clean CSV: " + new File(CLEAN_CSV_PATH).getAbsolutePath());
            System.out.println("ARFF file: " + new File(ARFF_PATH).getAbsolutePath());

            System.out.println("\nNote: Missing feature values are retained.");
            System.out.println("They should be imputed inside each training fold");
            System.out.println("during model evaluation to avoid data leakage.");

        } catch (Exception e) {
            System.out.println("\nData processing failed.");
            e.printStackTrace();
        }
    }

    // Save the prepared dataset as CSV
    private static void saveAsCsv(Instances dataset, String outputPath)
            throws Exception {

        if (dataset == null || dataset.numInstances() == 0) {
            throw new IllegalArgumentException(
                    "Dataset is empty or null."
            );
        }

        File outputFile = new File(outputPath);
        File parentDirectory = outputFile.getParentFile();

        if (parentDirectory != null && !parentDirectory.exists()) {
            if (!parentDirectory.mkdirs()) {
                throw new Exception(
                        "Unable to create output directory: "
                                + parentDirectory.getAbsolutePath()
                );
            }
        }

        CSVSaver saver = new CSVSaver();
        saver.setInstances(dataset);
        saver.setFile(outputFile);
        saver.writeBatch();

        System.out.println("Clean CSV saved successfully!");
        System.out.println("Output path: " + outputFile.getAbsolutePath());
        System.out.println("Number of instances: " + dataset.numInstances());
        System.out.println("Number of attributes: " + dataset.numAttributes());
    }
}
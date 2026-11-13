package heartdisease.Preprocessing;

import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import java.io.File;

public class DataLoader {
    
    //Set dataset path
    private static final String DATASET_PATH = "data/raw/heart_disease.csv";

    // Select target attribute
    private static final String TARGET_ATTRIBUTE = "Heart Disease Status";

    public static void main(String[] args) {
        try {

            // Load dataset
            Instances dataset = loadDataset(DATASET_PATH);

            // Dataset overview
            printDatasetOverview(dataset);

            // Attribute information
            printAttributeInfo(dataset);

            // Numerical analysis
            printNumericalAnalysis(dataset);

            // Categorical analysis
            printCategoricalAnalysis(dataset);

            // Missing value summary
            printMissingValue(dataset);

            // Target analysis
            printTargetAnalysis(dataset);

        } catch (Exception e){
            System.out.println("Error: Unable to analyze dataset");
            
            e.printStackTrace();
        }
    }


    // Method load dataset
    public static Instances loadDataset(String filePath) throws Exception{
        File file = new File(filePath);             // create a file represents for the file csv 

        // check if a file path exists
        if(!file.exists()){
            throw new Exception("Dataset file not found: " + filePath);
        }

        CSVLoader loader = new CSVLoader();         // create a object that read the file csv

        loader.setSource(file);             // pass the file to the loader

        Instances dataset = loader.getDataSet();        // Read the file

        System.out.println("Dataset loaded successfully");
    
        return dataset;
    }

    // Dataset overview
    public static void printDatasetOverview(Instances dataset){

        System.out.println("--- DATASET OVERVIEW ---");

        System.out.println("Dataset name: Heart Disease Dataset");

        System.out.println("Total records: " + dataset.numInstances());

        System.out.println("Total attributes: " + dataset.numAttributes());
    }

    // Attribute information
    public static void printAttributeInfo(Instances dataset){

        System.out.println("--- ATTRIBUTE INFORMATION");

        for (int i = 0; i < dataset.numAttributes(); i++){
            
            Attribute attribute = dataset.attribute(i);         // Take each attribute from data

            String type;

            // check type of attribute
            if (attribute.isNumeric()){
                type = "Numeric";
            } else if (attribute.isNominal()){
                type = "Categorical";
            } else if (attribute.isString()){
                type = "String";
            } else if (attribute.isDate()){
                type = "Date";
            } else {
                type = "Other";
            }

            String role;

            // Assign role to attribute
            if (attribute.name().equals(TARGET_ATTRIBUTE)) {
                role = "TARGET";
            } else {
                role = "FEATURE";
            }
            
            System.out.printf("%-25s | %-12s | %s%n", attribute.name(), type, role);
        }
    }


    // Numerical Analysis
    public static void printNumericalAnalysis(Instances dataset){
        
        System.out.println("--- NUMERICAL ANALYSIS ---");

        for (int i = 0; i < dataset.numAttributes(); i++){
            
            Attribute attribute = dataset.attribute(i);

            // check if attribute is Numeric
            if (!attribute.isNumeric()){
                continue;
            }

            AttributeStats stats = dataset.attributeStats(i);

            System.out.println("\nAttribute: " + attribute.name());

            // Count number of instances in attribute
            System.out.println("Count: " + (dataset.numInstances() - stats.missingCount));

            // Calculate mean of attribute
            System.out.printf("Mean: %.3f%n",stats.numericStats.mean);

            // Calculate standard deviation of attribute
            System.out.printf("Standard Deviation: %.3f%n",stats.numericStats.stdDev);

            // Calculate min  
            System.out.printf("Minimum: %.3f%n",stats.numericStats.min);

            // Calculate max
            System.out.printf("Maximum: %.3f%n",stats.numericStats.max);
        }
    }

    // Categorical analysis
    public static void printCategoricalAnalysis(Instances dataset){
        
        System.out.println("--- CATEGORICAL ANALYSIS ---");

        for (int i = 0; i < dataset.numAttributes(); i++){
            
            Attribute attribute = dataset.attribute(i);      // current attribute

            // check if attribute is Nomial
            if (!attribute.isNominal()){
                continue;
            }

            AttributeStats stats = dataset.attributeStats(i);       // retrieve statistics for attribute

            System.out.println("\nAttribute: " + attribute.name());

            // iterate values of Nominal attribute
            for (int j = 0; j < attribute.numValues(); j++){

                String value = attribute.value(j);          // current value Nominal

                int count = stats.nominalCounts[j];         // count values

                System.out.printf("%-15s : %d%n", value, count);
            }
        }
    }

    //Missing value
    public static void printMissingValue(Instances dataset){

        System.out.println("--- MISSING VALUE SUMMARY ---");

        int totalMissing = 0;

        for (int i = 0; i < dataset.numAttributes(); i++){
            
            AttributeStats stats = dataset.attributeStats(i);

            int missing = stats.missingCount;           // number of missing values for each attribute

            totalMissing += missing;

            if (missing > 0){
                double percentage = (double) missing / dataset.numInstances() * 100;

                System.out.printf("%-25s : %d (%.2f%%)%n", dataset.attribute(i).name(), missing, percentage);
            }
        }

        System.out.println("\nTotal Missing Values: " + totalMissing);
    }

    // Target analysis
    public static void printTargetAnalysis(Instances dataset){

        System.out.println("--- TARGET ANALYSIS ---");

        Attribute target = dataset.attribute(TARGET_ATTRIBUTE);
        
        // check if target attribute exists
        if (target == null){
            System.out.println("Target attribute not found");
            return;
        }

        // check type value of target attribute
        if(!target.isNominal()){
            System.out.println("Target must be categorical");
            return;
        }

        int targetIndex = target.index();           // take postion index of target

        AttributeStats stats = dataset.attributeStats(targetIndex);         //  // retrieve statistics

        System.out.println("Target Attribute: " + target.name());

        int validCount = dataset.numInstances() - stats.missingCount;       // calculate number of valid records

        for (int i = 0; i < target.numValues(); i++){
            
            String className = target.value(i);

            int count = stats.nominalCounts[i];

            double percentage;

            if (validCount == 0){
                    percentage = 0;
            } else {
                percentage = (double) count / validCount * 100;
            }

             System.out.printf("%-15s : %d (%.2f%%)%n", className, count, percentage);
        }
    }
}

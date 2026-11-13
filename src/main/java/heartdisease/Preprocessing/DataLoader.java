package heartdisease.Preprocessing;

import weka.core.Attribute;
import weka.core.AttributeStats;
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

        System.out.println("\n--- ATTRIBUTE INFORMATION");

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

}

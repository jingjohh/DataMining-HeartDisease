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

}

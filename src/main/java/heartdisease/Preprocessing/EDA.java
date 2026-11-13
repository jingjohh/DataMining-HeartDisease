package heartdisease.Preprocessing;

import weka.core.Instances;
import weka.core.Attribute;
import java.util.HashSet;
import java.util.Set;

public class EDA {

    // Find Missing value
    public static void printMissingValues(Instances dataset){
        System.out.println("\n--- MISSING VALUE ANALYSIS ---");

        int totalRecords = dataset.numInstances();
        int totalMissing = 0;

        if (totalRecords == 0) {
            System.out.println("Dataset is empty!");
            return;
        }

        // Analyze missing values of each attribute
        for (int i = 0; i < dataset.numAttributes(); i++) {

            Attribute attribute = dataset.attribute(i);

            int missingCount = 0;

            // Count missing values
            for (int j = 0; j < totalRecords; j++) {

                if (dataset.instance(j).isMissing(i)) {
                    missingCount++;
                }
            }

            int validCount = totalRecords - missingCount;

            double missingPercentage = (double) missingCount / totalRecords * 100;

            totalMissing += missingCount;

            System.out.println(
                    attribute.name()
                    + " | Missing: " + missingCount
                    + " | Valid: " + validCount
                    + " | Missing Rate: "
                    + String.format("%.2f", missingPercentage) + "%"
            );
        }

        System.out.println("Total Records: " + totalRecords);
        System.out.println("Total Missing Values: " + totalMissing);
    }

    // Duplicate Analysis
    public static void printDuplicateAnalysis(Instances dataset){

        System.out.println("\n--- DUPLICATE ANALYSIS ---");

        Set<String> uniqueRecords = new HashSet<>();        // create a set to store unique records

        int duplicateCount = 0;
        
        // iterate all record of dataset
        for (int i = 0; i < dataset.numInstances(); i++){

            String record = dataset.instance(i).toString();     // convert record to string

            // check the current record has appeared previously
            if (uniqueRecords.contains(record)){
                duplicateCount++;
            } else{
                uniqueRecords.add(record);
            }
        }

        System.out.println("Total Records: " + dataset.numInstances());
        System.out.println("Unique Records: " + uniqueRecords.size());
        System.out.println("Duplicate Records: " + duplicateCount);

    }
    
}

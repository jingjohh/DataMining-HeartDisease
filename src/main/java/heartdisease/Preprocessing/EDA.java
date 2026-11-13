package heartdisease.Preprocessing;

import weka.core.Instances;
import weka.core.Attribute;
import java.util.HashSet;
import java.util.Set;
import java.util.ArrayList;
import java.util.Collections;
import weka.core.AttributeStats;

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
    
    // Numerical Analysis

    public static void printNumericalAnalysis(Instances dataset){

        System.out.println("\n--- NUMERICAL ANALYSIS ---");

        for (int i = 0; i < dataset.numAttributes(); i++){

            Attribute attribute = dataset.attribute(i);

            if (!attribute.isNumeric()){
                continue;
            }

            AttributeStats stats = dataset.attributeStats(i);

            // Collect valid numerical values
            ArrayList<Double> values = new ArrayList<>();

            for (int j = 0; j < dataset.numInstances(); j++){

                if (!dataset.instance(j).isMissing(i)){
                    values.add(dataset.instance(j).value(i));
                }
            }

            // Sort values in ascending order
            Collections.sort(values);

            int count = values.size();

            if (count == 0){
                System.out.println("\nAttribute: " + attribute.name());
                System.out.println("No valid numerical values");
                continue;
            }

            // Calculate Median
            int middle = count / 2;
            double median;

            if (count % 2 == 1){
                median = values.get(middle);
            } else {
                median = (values.get(middle - 1)
                        + values.get(middle)) / 2.0;
            }

            // Calculate Q1
            double q1;
            int lowerSize = middle;

            if (lowerSize % 2 == 1){
                q1 = values.get(lowerSize / 2);
            } else {
                q1 = (values.get(lowerSize / 2 - 1)
                        + values.get(lowerSize / 2)) / 2.0;
            }

            // Calculate Q3
            double q3;
            int upperStart = (count + 1) / 2;
            int upperSize = count - upperStart;

            if (upperSize % 2 == 1){
                q3 = values.get(upperStart + upperSize / 2);
            } else {
                q3 = (values.get(upperStart + upperSize / 2 - 1)
                        + values.get(upperStart + upperSize / 2)) / 2.0;
            }

            // Calculate IQR
            double iqr = q3 - q1;

            System.out.println("\nAttribute: " + attribute.name());
            System.out.println("Count: " + count);
            
            System.out.printf("Mean: %.3f%n", stats.numericStats.mean);
            System.out.printf("Median (50%%): %.3f%n", median);
            System.out.printf("Standard Deviation: %.3f%n", stats.numericStats.stdDev);
            
            System.out.printf("Minimum: %.3f%n", stats.numericStats.min);
            System.out.printf("Maximum: %.3f%n", stats.numericStats.max);
            
            System.out.printf("Q1 (25%%): %.3f%n", q1);
            System.out.printf("Q3 (75%%): %.3f%n", q3);
            System.out.printf("IQR: %.3f%n", iqr);
        }
    }


}

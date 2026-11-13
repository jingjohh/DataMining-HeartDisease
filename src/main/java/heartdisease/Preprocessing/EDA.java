package heartdisease.Preprocessing;

import weka.core.Instances;
import weka.core.Attribute;
import java.util.HashSet;
import java.util.Set;
import java.util.ArrayList;
import java.util.Collections;
import weka.core.AttributeStats;

public class EDA {

    private static final String TARGET_ATTRIBUTE = "Heart Disease Status";

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

            System.out.println("\nAttribute: " + attribute.name());
            System.out.println("Count: " + count);
            
            System.out.printf("Mean: %.3f%n", stats.numericStats.mean);
            System.out.printf("Median (50%%): %.3f%n", median);
            System.out.printf("Standard Deviation: %.3f%n", stats.numericStats.stdDev);
            
            System.out.printf("Minimum: %.3f%n", stats.numericStats.min);
            System.out.printf("Maximum: %.3f%n", stats.numericStats.max);
            
        }
    }

    // Categorical analysis
    public static void printCategoricalAnalysis(Instances dataset){
        
        System.out.println("\n--- CATEGORICAL ANALYSIS ---");

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

    // Target analysis
    public static void printTargetAnalysis(Instances dataset){


        System.out.println("\n--- TARGET ANALYSIS ---");

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

        int validCount = dataset.numInstances() - stats.missingCount;      // calculate number of valid records

        if (validCount == 0){
            System.out.println("No valid target values");
            return;
        }

        int majorityCount = 0;
        int minorityCount = validCount;

        String majorityClass = "";
        String minorityClass = "";

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
            
            // Identify majority and minority classes
            if (count > majorityCount){
                majorityCount = count;
                majorityClass = className;
            }

            if (count < minorityCount){
                minorityCount = count;
                minorityClass = className;
            }
        }

        System.out.println("Majority Class: " + majorityClass + " (" + majorityCount + ")");

        System.out.println("Minority Class: " + minorityClass + " (" + minorityCount + ")");

        if (minorityCount > 0){

            double imbalanceRatio = (double) majorityCount / minorityCount;

            System.out.printf("Imbalance Ratio: %.2f : 1%n", imbalanceRatio);
        } else {
            System.out.println("Imbalance Ratio: Undefined (a class has zero records)");
        }
        
    }

    // Outlier

    public static void printOutlierAnalysis(Instances dataset){

        System.out.println("\n--- OUTLIER ANALYSIS ---");

        for (int i = 0; i < dataset.numAttributes(); i++){

            Attribute attribute = dataset.attribute(i);

            // Only analyze numerical attributes
            if (!attribute.isNumeric()){
                continue;
            }

            ArrayList<Double> values = new ArrayList<>();

            // Collect valid values
            for (int j = 0; j < dataset.numInstances(); j++){

                if (!dataset.instance(j).isMissing(i)){
                    values.add(dataset.instance(j).value(i));
                }
            }

            Collections.sort(values);

            int count = values.size();

            if (count < 2){
                System.out.println("\nAttribute: " + attribute.name());
                System.out.println("Not enough valid values");
                continue;
            }

            // Calculate Q1 and Q3 using median of halves
            int middle = count / 2;

            int upperStart = (count + 1) / 2;

            double q1;
            int lowerSize = middle;

            if (lowerSize % 2 == 1){
                q1 = values.get(lowerSize / 2);
            } else if (lowerSize > 0){
                q1 = (values.get(lowerSize / 2 - 1)
                        + values.get(lowerSize / 2)) / 2.0;
            } else {
                q1 = values.get(0);
            }

            int upperSize = count - upperStart;

            double q3;

            if (upperSize % 2 == 1){
                q3 = values.get(upperStart + upperSize / 2);
            } else {
                q3 = (values.get(upperStart + upperSize / 2 - 1)
                        + values.get(upperStart + upperSize / 2)) / 2.0;
            }

            // Calculate IQR and boundaries
            double iqr = q3 - q1;

            double lowerBound = q1 - 1.5 * iqr;
            double upperBound = q3 + 1.5 * iqr;

            int outlierCount = 0;

            // Count outliers
            for (double value : values){

                if (value < lowerBound || value > upperBound){
                    outlierCount++;
                }
            }

            double outlierPercentage =
                    (double) outlierCount / count * 100;

            System.out.println("\nAttribute: " + attribute.name());
            System.out.printf("Q1: %.3f%n", q1);
            System.out.printf("Q3: %.3f%n", q3);
            System.out.printf("IQR: %.3f%n", iqr);
            System.out.printf("Lower Bound: %.3f%n", lowerBound);
            System.out.printf("Upper Bound: %.3f%n", upperBound);
            System.out.println("Outlier Count: " + outlierCount);
            System.out.printf("Outlier Rate: %.2f%%%n", outlierPercentage);
        }
    }

    // Pearson Correlation
    public static double calculateCorrelation(Instances dataset, int firstIndex, int secondIndex){

        double sumX = 0;
        double sumY = 0;
        double sumXY = 0;
        double sumX2 = 0;
        double sumY2 = 0;

        int count = 0;

        for (int i = 0; i < dataset.numInstances(); i++){

            if (dataset.instance(i).isMissing(firstIndex) || dataset.instance(i).isMissing(secondIndex)){
                continue;
            }

            double x = dataset.instance(i).value(firstIndex);
            double y = dataset.instance(i).value(secondIndex);

            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
            sumY2 += y * y;

            count++;
        }

        if (count < 2){
            return Double.NaN;
        }

        double numerator = count * sumXY - sumX * sumY;

        double denominatorX = count * sumX2 - sumX * sumX;

        double denominatorY = count * sumY2 - sumY * sumY;

        double denominator = Math.sqrt(denominatorX * denominatorY);

        if (denominator == 0){
            return Double.NaN;
        }

        return numerator / denominator;
    }

    // Correlation matrix
    public static void printCorrelationAnalysis(Instances dataset){

        System.out.println("\n--- CORRELATION ANALYSIS ---");

        ArrayList<Integer> numericalIndices = new ArrayList<>();

        // Collect numerical features, excluding target
        for (int i = 0; i < dataset.numAttributes(); i++){

            Attribute attribute = dataset.attribute(i);

            if (attribute.isNumeric()
                    && !attribute.name().equals(TARGET_ATTRIBUTE)){

                numericalIndices.add(i);
            }
        }

        if (numericalIndices.isEmpty()){
            System.out.println("No numerical features found");
            return;
        }

        // Print correlation matrix header
        System.out.printf("%-20s", "");

        for (int index : numericalIndices){
            System.out.printf("%-12s",
                    dataset.attribute(index).name());
        }

        System.out.println();

        // Calculate correlation matrix
        for (int firstIndex : numericalIndices){

            System.out.printf("%-20s",
                    dataset.attribute(firstIndex).name());

            for (int secondIndex : numericalIndices){

                double correlation = calculateCorrelation(
                        dataset, firstIndex, secondIndex
                );

                if (Double.isNaN(correlation)){
                    System.out.printf("%-12s", "N/A");
                } else {
                    System.out.printf("%-12.3f", correlation);
                }
            }

            System.out.println();
        }

        // Identify highly correlated feature pairs
        System.out.println("\n--- HIGH CORRELATION PAIRS ---");

        boolean found = false;

        for (int i = 0; i < numericalIndices.size(); i++){

            for (int j = i + 1; j < numericalIndices.size(); j++){

                int firstIndex = numericalIndices.get(i);
                int secondIndex = numericalIndices.get(j);

                double correlation = calculateCorrelation(
                        dataset, firstIndex, secondIndex
                );

                if (!Double.isNaN(correlation)
                        && Math.abs(correlation) >= 0.7){

                    System.out.printf(
                            "%s - %s : %.3f%n",
                            dataset.attribute(firstIndex).name(),
                            dataset.attribute(secondIndex).name(),
                            correlation
                    );

                    found = true;
                }
            }
        }

        if (!found){
            System.out.println("No feature pairs with |r| >= 0.7");
        }
    }


}

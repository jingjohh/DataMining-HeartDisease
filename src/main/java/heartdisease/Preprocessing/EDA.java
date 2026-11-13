package heartdisease.Preprocessing;

import weka.core.Instances;
import weka.core.Attribute;

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
}

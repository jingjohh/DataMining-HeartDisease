
package heartdisease;

import java.io.StringReader;
import java.util.Random;

import weka.core.Instances;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;

public class Main {

    public static void main(String[] args) throws Exception {

        // 1. Create a small sample dataset
        String arff = """
            @relation heart_disease_demo

            @attribute age numeric
            @attribute cholesterol {normal,high}
            @attribute smoking {no,yes}
            @attribute heart_disease {no,yes}

            @data
            25,normal,no,no
            30,normal,yes,no
            35,high,no,no
            40,normal,no,no
            45,high,yes,yes
            50,normal,yes,no
            55,high,yes,yes
            60,high,no,yes
            65,high,yes,yes
            70,normal,no,yes
            75,high,yes,yes
            80,high,no,yes
            """;

        // 2. Load data into Weka
        Instances data = new Instances(new StringReader(arff));
        data.setClassIndex(data.numAttributes() - 1);

        // 3. Initialize and train J48
        J48 model = new J48();
        model.buildClassifier(data);

        System.out.println("J48 model trained successfully!");

        // 4. Evaluate using 3-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 3, new Random(42));

        // 5. Print results
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }
}

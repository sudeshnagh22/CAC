import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.util.Random;

public class PlayerClassification {

    /**
     * Main method to run the machine learning example.
     * @param args Command-line arguments (not used).
     */
    public static void main(String[] args) {
        try {
            // Load dataset
            DataSource source = new DataSource("players.csv");
            Instances data = source.getDataSet();
            
            // Set the class index (the attribute to predict)
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Build classifier
            J48 tree = new J48();
            tree.buildClassifier(data);

            // Evaluate classifier
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(tree, data, 10, new Random(1));

            // Output evaluation
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            System.out.println("Confusion Matrix:");
            for (double[] row : eval.confusionMatrix()) {
                for (double value : row) {
                    System.out.print(value + " ");
                }
                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

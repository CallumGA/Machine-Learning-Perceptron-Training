/*
 * Perceptron - Machine Learning, Assignment 5 - Part 2.2
 *
 * Callum Anderson
 * T00708915
 * October 20, 2024
 *
 * */


class Forward_Propagation {
    // defines the inputs and outputs as double arrays in the code
    double[][] inputs = new double[][]{

            {0, 0, 1},

            {1, 1, 1},

            {1, 0, 1},

            {0, 1, 1}
    };
    double[][] outputs = new double[][]{

            {0},

            {1},

            {1},

            {0}
    };
    // weights and bias
    private double[] weights;
    private double bias;
    private double learningRate;
    private Integer numberOfFeatures = 0;
    public Forward_Propagation(Integer numberOfFeatures) {
        this.weights = new double[numberOfFeatures];
        this.numberOfFeatures = numberOfFeatures;
    }

    // sigmoid function calculation
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // generates the initial weights and bias using Math.random().
    public double[] GenerateInitialWeights() {
        // initialize weights randomly
        for (int i = 0; i < numberOfFeatures; i++) {
            weights[i] = (Math.random() * 2) - 1;
        }
        return weights;
    }

    public double CalculateError() {
        /*
            1.	Error Calculation:
            After the perceptron makes a prediction, it compares that prediction to the true label (the expected output from the training data). If the prediction is incorrect, the perceptron calculates an error, which is simply:

            error = label - prediction

            // *** for use the label will be the Alarm column ***

            The error tells the perceptron how far off its prediction was. If the perceptron under-predicts (e.g., it predicts 0 when the label is 1), the error is positive. If it over-predicts (e.g., it predicts 1 when the label is 0), the error is negative.
        */
        return 0;
    }

    public double UpdateWeights() {
        return 0;
    }

    // calculates the outputs of a given input set.
    public String GenerateOutput() {
        return null;
    }
}

class Train_Test {

}


public class Main {
    public static void main(String[] args) {

        // we have 3 inputs which will require 3 weights
        var weights = new Forward_Propagation(3);
        weights.GenerateInitialWeights();
        System.out.println();
    }
}
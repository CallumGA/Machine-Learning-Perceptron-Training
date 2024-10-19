/*
 * Perceptron - Machine Learning, Assignment 5 - Part 2.2
 *
 * Callum Anderson
 * T00708915
 * October 20, 2024
 *
 * */

// FORWARD PROPAGATION
// Steps:
// 1. input data (we have 3 inputs)
// 2. calculate weighted sum via weights and bias
// 3. run weighted sum through sigmoid (activation function)

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
    private double bias = 0.0;
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


    // calculates the outputs of a given input set based on the weights and bias
    public double GenerateOutput() {
        double sum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            // TODO: we will need a nested for-loop for the nested array
            // sum += weights[i] * inputs[i];
        }
        // add bias
        sum += bias;
        // apply the sigmoid function to get the prediction
        return sigmoid(sum);
    }
}

// BACKWARD PROPAGATION
// Steps:
// 1. calculate the error by comparing the predicted output from forward propagation with the true label: error = label - prediction
//    the error tells the perceptron how far off its prediction was.
// 2. we calculate the new weights with the new weights based on the following rule: new_weight = current_weight + (learning rate x error calc x input)
// 3. we calculate the new bias: bias = bias + (learning rate x error)
// 4. we actually update the weights with the new weights and bias with new bias
class Train_Test {

    private double learningRate;

    public Train_Test() {

    }

    public double CalculateError() {
        return 0;
    }

}


public class Main {
    public static void main(String[] args) {

        // we have 3 inputs which will require 3 weights
        var weights = new Forward_Propagation(3);
        weights.GenerateInitialWeights();
        System.out.println();
    }
}
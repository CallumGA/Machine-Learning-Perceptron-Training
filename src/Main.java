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

    // calculates the outputs for a given 2D input set based on the weights and bias
    public double[] GenerateOutput() {
        double[] outputs = new double[inputs.length];
        // iterate over each input row in the 2D array
        for (int i = 0; i < inputs.length; i++) {
            double sum = 0.0;
            // calculate the weighted sum for the current input row
            for (int j = 0; j < weights.length; j++) {
                // multiply each input by its corresponding weight
                sum += weights[j] * inputs[i][j];
            }
            // add bias to the sum
            sum += bias;
            // apply the sigmoid function
            outputs[i] = sigmoid(sum);
        }
        return outputs;
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

        // generate initial weights
        var forwardPropagation = new Forward_Propagation(3);
        forwardPropagation.GenerateInitialWeights();

        // generate output for the forward propagation
        var forwardPassOutputs = forwardPropagation.GenerateOutput();

        System.out.println();
    }
}
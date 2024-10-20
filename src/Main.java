/*
 * Perceptron - Machine Learning, Assignment 5 - Part 2.2
 *
 * Callum Anderson
 * T00708915
 * October 20, 2024
 *
 * */

// FORWARD PROPAGATION CLASS
class Forward_Propagation {
    // weights and bias for the perceptron
    public double[] weights;
    public double bias = 0.0;
    // sensor input
    double[][] inputs = new double[][]{
            {0, 0, 1},  // Instance 1
            {1, 1, 1},  // Instance 2
            {1, 0, 1},  // Instance 3
            {0, 1, 1}   // Instance 4
    };
    // labels for what is the expected classification
    double[] labels = new double[]{
            // fake
            0,
            // real
            1,
            // real
            1,
            // fake
            0
    };
    private int numberOfFeatures = 0;

    public Forward_Propagation(int numberOfFeatures) {
        this.weights = new double[numberOfFeatures];
        this.numberOfFeatures = numberOfFeatures;
    }

    // our activation function (sigmoid)
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // derivitive of sigmoid used in back propagation
    public static double sigmoidDerivative(double output) {
        return output * (1 - output);
    }

    // randomly creates our initial weights (pre-trained)
    public double[] GenerateInitialWeights() {
        for (int i = 0; i < numberOfFeatures; i++) {
            weights[i] = (Math.random() * 2) - 1;  // Random weights between -1 and 1
        }
        return weights;
    }

    // calculates the outputs for a given 2D input set based on the weights and bias
    public double[] GenerateOutput() {
        double[] outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < weights.length; j++) {
                // weighted input sums
                sum += weights[j] * inputs[i][j];
            }
            sum += bias;
            outputs[i] = sigmoid(sum);
        }
        return outputs;
    }

    // classifies a new input using the trained weights and bias
    public double classify(double[] input) {
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            sum += weights[i] * input[i];
        }
        sum += bias;
        double output = sigmoid(sum);

        // if output is greater than 0.5 classify as real, if less then classify as fake
        return output >= 0.5 ? 1 : 0;
    }

    // actually update the weights in our model based on the results of the back propagation
    public void updateWeightsAndBias(double[] updatedWeights, double updatedBias) {
        this.weights = updatedWeights;
        this.bias = updatedBias;
    }
}

// BACKWARD PROPAGATION CLASS
class Train_Test {

    public double[] weights;
    private double[] forwardPropOutputs;
    private double[][] inputs;
    private double bias;

    public Train_Test(double[] forwardPropagationOutputs, double[][] inputs, double[] weights, double bias) {
        this.forwardPropOutputs = forwardPropagationOutputs;
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
    }

    // calculate the error for each output via: (label - prediction)
    public double[] CalculateError(double[] labels) {
        double[] errorCalculations = new double[forwardPropOutputs.length];
        for (int i = 0; i < forwardPropOutputs.length; i++) {
            errorCalculations[i] = labels[i] - forwardPropOutputs[i];
        }
        return errorCalculations;
    }

    // update weights and bias using the perceptron learning rule, incorporating the sigmoid derivative
    public void UpdateWeightsBias(double[] errors) {
        // this is our learning rate
        double learningRate = 0.1;
        // update weights based on the error and the corresponding input
        for (int i = 0; i < weights.length; i++) {
            double weightUpdate = 0.0;
            for (int j = 0; j < inputs.length; j++) {
                double output = forwardPropOutputs[j];
                // apply the learning rule: weights[i] += learning_rate * error * sigmoid_derivative * input
                weightUpdate += learningRate * errors[j] * Forward_Propagation.sigmoidDerivative(output) * inputs[j][i];
            }
            // actually adjust our weights
            weights[i] += weightUpdate;
        }

        // update the bias
        double biasUpdate = 0.0;
        for (int i = 0; i < errors.length; i++) {
            biasUpdate += learningRate * errors[i] * Forward_Propagation.sigmoidDerivative(forwardPropOutputs[i]);
        }
        bias += biasUpdate;
    }

    // getter for updated weights
    public double[] getUpdatedWeights() {
        return weights;
    }

    // getter for updated bias
    public double getUpdatedBias() {
        return bias;
    }
}

// MAIN CLASS
public class Main {
    public static void main(String[] args) {

        // create a forward propagation object with 3 features (inputs)
        Forward_Propagation forwardPropagation = new Forward_Propagation(3);
        // generate the initial random weights
        forwardPropagation.GenerateInitialWeights();

        // we run through 1000 iterations of training
        int numIterations = 1000;

        for (int iteration = 0; iteration < numIterations; iteration++) {
            // we do the forward propagation to get outputs
            double[] forwardPropOutputs = forwardPropagation.GenerateOutput();

            // we run backpropagation
            Train_Test backwardPropagation = new Train_Test(forwardPropOutputs, forwardPropagation.inputs, forwardPropagation.weights, forwardPropagation.bias);
            double[] errorCalcs = backwardPropagation.CalculateError(forwardPropagation.labels);

            // update the weights and bias based on errors
            backwardPropagation.UpdateWeightsBias(errorCalcs);

            // feed updated weights and bias back to forward propagation
            forwardPropagation.updateWeightsAndBias(backwardPropagation.getUpdatedWeights(), backwardPropagation.getUpdatedBias());
        }

        // final outputs
        System.out.println("\n");
        System.out.println("Final outputs after 1000 iterations: " + java.util.Arrays.toString(forwardPropagation.GenerateOutput()));
        System.out.println("Final weights: " + java.util.Arrays.toString(forwardPropagation.weights));
        System.out.println("Final bias: " + forwardPropagation.bias);

        // feed in the unseen instance to see if it classifies correctly
        double[] newInstance = new double[]{0, 0, 0};
        double classificationResult = forwardPropagation.classify(newInstance);
        System.out.println("Classification for new instance {0, 0, 0}: " + (classificationResult == 1 ? "Real" : "Fake"));
    }
}
// FORWARD PROPAGATION CLASS
class Forward_Propagation {
    // Weights and bias for the perceptron
    public double[] weights;
    public double bias = 0.0;
    // Defines the inputs and expected outputs as double arrays
    double[][] inputs = new double[][]{
            {0, 0, 1},  // Instance 1
            {1, 1, 1},  // Instance 2
            {1, 0, 1},  // Instance 3
            {0, 1, 1}   // Instance 4
    };
    // Labels for the expected outputs (Alarm classification: 0 for Fake, 1 for Real)
    double[] labels = new double[]{
            0,  // Fake (Instance 1)
            1,  // Real (Instance 2)
            1,  // Real (Instance 3)
            0   // Fake (Instance 4)
    };
    private int numberOfFeatures = 0;

    // Constructor to initialize weights array
    public Forward_Propagation(int numberOfFeatures) {
        this.weights = new double[numberOfFeatures];
        this.numberOfFeatures = numberOfFeatures;
    }

    // Sigmoid activation function
    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Sigmoid derivative function (used for backpropagation)
    public static double sigmoidDerivative(double output) {
        return output * (1 - output);
    }

    // Generates the initial weights and bias using Math.random().
    public double[] GenerateInitialWeights() {
        for (int i = 0; i < numberOfFeatures; i++) {
            weights[i] = (Math.random() * 2) - 1;  // Random weights between -1 and 1
        }
        return weights;
    }

    // Calculates the outputs for a given 2D input set based on the weights and bias
    public double[] GenerateOutput() {
        double[] outputs = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            double sum = 0.0;
            for (int j = 0; j < weights.length; j++) {
                sum += weights[j] * inputs[i][j];  // Weighted sum of inputs
            }
            sum += bias;  // Adding bias
            outputs[i] = sigmoid(sum);  // Applying sigmoid activation
        }
        return outputs;
    }

    // Classifies a new input using the trained weights and bias
    public double classify(double[] input) {
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            sum += weights[i] * input[i];
        }
        sum += bias;
        double output = sigmoid(sum);  // Apply sigmoid to the sum

        // If output >= 0.5, classify as Real (1), otherwise Fake (0)
        return output >= 0.5 ? 1 : 0;
    }

    // Updates weights and bias after backpropagation
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

    // Constructor to set initial values from forward propagation
    public Train_Test(double[] forwardPropagationOutputs, double[][] inputs, double[] weights, double bias) {
        this.forwardPropOutputs = forwardPropagationOutputs;
        this.inputs = inputs;
        this.weights = weights;
        this.bias = bias;
    }

    // Calculate the error for each output: (label - prediction)
    public double[] CalculateError(double[] labels) {
        double[] errorCalculations = new double[forwardPropOutputs.length];
        for (int i = 0; i < forwardPropOutputs.length; i++) {
            errorCalculations[i] = labels[i] - forwardPropOutputs[i];
        }
        return errorCalculations;
    }

    // Update weights and bias using the perceptron learning rule, incorporating the sigmoid derivative
    public void UpdateWeightsBias(double[] errors) {
        double learningRate = 0.1;  // Learning rate for weight updates

        // Update weights based on the error and the corresponding input
        for (int i = 0; i < weights.length; i++) {
            double weightUpdate = 0.0;
            for (int j = 0; j < inputs.length; j++) {
                double output = forwardPropOutputs[j];
                // Apply the learning rule: weights[i] += learning_rate * error * sigmoid_derivative * input
                weightUpdate += learningRate * errors[j] * Forward_Propagation.sigmoidDerivative(output) * inputs[j][i];
            }
            weights[i] += weightUpdate;  // Adjust weight
        }

        // Update bias based on the errors and sigmoid derivative
        double biasUpdate = 0.0;
        for (int i = 0; i < errors.length; i++) {
            biasUpdate += learningRate * errors[i] * Forward_Propagation.sigmoidDerivative(forwardPropOutputs[i]);
        }
        bias += biasUpdate;  // Adjust bias
    }

    // Getter for updated weights
    public double[] getUpdatedWeights() {
        return weights;
    }

    // Getter for updated bias
    public double getUpdatedBias() {
        return bias;
    }
}

// MAIN CLASS
public class Main {
    public static void main(String[] args) {

        // Create a forward propagation object with 3 features (inputs)
        Forward_Propagation forwardPropagation = new Forward_Propagation(3);
        forwardPropagation.GenerateInitialWeights();  // Generate random initial weights

        // Set the number of iterations for training (1000)
        int numIterations = 1000;

        // Training loop for 1000 iterations
        for (int iteration = 0; iteration < numIterations; iteration++) {
            // Perform forward propagation to get the outputs
            double[] forwardPropOutputs = forwardPropagation.GenerateOutput();

            // Create a backpropagation object to calculate error and update weights
            Train_Test backwardPropagation = new Train_Test(forwardPropOutputs, forwardPropagation.inputs, forwardPropagation.weights, forwardPropagation.bias);
            double[] errorCalcs = backwardPropagation.CalculateError(forwardPropagation.labels);

            // Update the weights and bias based on errors
            backwardPropagation.UpdateWeightsBias(errorCalcs);

            // Pass updated weights and bias back to forward propagation
            forwardPropagation.updateWeightsAndBias(backwardPropagation.getUpdatedWeights(), backwardPropagation.getUpdatedBias());
        }

        // Final output after training
        System.out.println("\n");
        System.out.println("Final outputs after 1000 iterations: " + java.util.Arrays.toString(forwardPropagation.GenerateOutput()));
        System.out.println("Final weights: " + java.util.Arrays.toString(forwardPropagation.weights));
        System.out.println("Final bias: " + forwardPropagation.bias);

        // Test with a new unseen instance {0, 0, 0} and classify it
        double[] newInstance = new double[]{0, 0, 0};
        double classificationResult = forwardPropagation.classify(newInstance);
        System.out.println("Classification for new instance {0, 0, 0}: " + (classificationResult == 1 ? "Real" : "Fake"));
    }
}
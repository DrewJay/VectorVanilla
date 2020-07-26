import {
    NodeGroup,
    TerminalReasons,
    ActivationFunctionsCollection,
    CostFunctionTypes,
    CostFunctionsCollection,
} from '../../common/structures/types.struct';

import {
    Getlib
} from '../../common/structures/general/decorators';

import {
    generalBackpropagation,
} from '../../common/lib/scientific.lib';
import { randNum } from '../../common/lib/utils.lib';

/**
 * Unit suited for data distribution across the neural network.
 */
@Getlib({
    activations: 'common/lib/functions/activations.lib',
    costs: 'common/lib/functions/costs.lib',
    derivatives: 'common/lib/functions/derivatives.lib',
})
export class DistributionUnit {
    /**
     * The output node value at the end of the iteration.
     */
    public output: number[] = [];

    /**
     * Training error.
     */
    public error: number = 0;

    /**
     * Provided input data.
     */
    public inputData: number[] = [];

    /**
     * Expected data.
     */
    public targetData: number[] = [];

    /**
     * Generated descriptive T2 layer object.
     */
    public layers: NodeGroup[];

    /**
     * Chosen error function.
     */
    private costFunction: CostFunctionTypes;

    /**
     * Input batch size.
     */
    private batchSize: number = 1;

    /**
     * Learning rate coefficient.
     */
    private learningRate: number = 1;

    /**
     * When on, iteration use learning rate adaptation.
     */
    private adaptive: boolean = false;

    /**
     * Deremine whether to track errors during iterations.
     */
    private errorTracking = false;

    /**
     * When learning rate is being modified, these are the rates.
     */
    private learningRateTuneRates = {
        min: .001,
        max: .009,
    };

    /**
     * Collection of activation functions.
     */
    private activations: ActivationFunctionsCollection;

    /**
     * Collection of cost functions.
     */
    private costs: CostFunctionsCollection;

    /**
     * Previous error correction action (up/down).
     */
    private errorCorrectionIncrease = false;

    /**
     * How many times error can increase in a row.
     */
    private errorFluctuationLimit = 1;

    /**
     * How many times error got worse in a row.
     */
    private errorCounter = 0;

    /**
     * Non-error counter.
     */
    private subsequentErorReducingIterations = 0;

    /**
     * Constant values used outside of this class during training.
     */
    public static constants = {
        RECOMMENDED_DIVERGENCE_THRESHOLD: 2,
        RECOMMENDED_TRAINING_CONFIDENCE: Math.pow(10, -30),
    };

    /**
     * Store layers reference locally and modify it on the fly.
     * 
     * @param layers - T2 layers object
     * @param costFunction - Error function to be used
     * @param batchSize - Training sample size
     * @param learningRate - The learning rate (min. 1)
     * @param adaptive - Whether to use internal adaptive methods
     * @param errorTracking - Whether to track errors during training
     */
    constructor(
        layers: NodeGroup[],
        costFunction: CostFunctionTypes,
        batchSize: number,
        learningRate: number = 1,
        adaptive: boolean = false,
        errorTracking: boolean = false
    ) {
        this.layers = layers;
        this.costFunction = costFunction;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
        this.adaptive = adaptive;
        this.errorTracking = errorTracking;
    }

    /**
     * Initialize data that will enter the network.
     * 
     * @param data - Input dataset
     */
    public initializeInputData(data: number[]) {
        this.inputData = data;
    }

    /**
     * Initialize seeked values.
     * 
     * @param data - Target dataset
     */
    public initializeTargetData(data: number[]) {
        this.targetData = data;
    }

    /**
     * Terminate training if particular conditions are fulfilled.
     * 
     * @param reasons - Reasons of termination
     * @returns Textual reson description
     */
    public terminateOn(reasons: TerminalReasons[]) {
        if (
            reasons.includes('divergence')
            && this.error >= DistributionUnit.constants.RECOMMENDED_DIVERGENCE_THRESHOLD
        ) {
            return 'Possible divergence detected.';
        } else if (
            reasons.includes('convergence')
            && this.error <= DistributionUnit.constants.RECOMMENDED_TRAINING_CONFIDENCE
        ) {
            return 'Training terminated for high confidence results.';
        }
    }

    /**
     * Distribute data across the neural network.
     * 
     * @param feedForwardOnly - Only feed the data in to get the output without any network
     * characteristics modifications (a.k.a "prediction")
     */
    public iterate(feedForwardOnly = false) {
        // Get input layer size and base increment on it.
        const inputLayerSize = this.layers[0].collection.length;

        for (let i = 0; i < this.inputData.length; i += inputLayerSize) {
            this.layers.forEach((layer) => {
                // Load first layer with input data.
                const input = layer.flags.includes('input');
                const output = layer.flags.includes('output');
                this.output = [];

                // Feed the input layer.
                if (input) {
                    layer.collection.forEach((inputNode, index) => {
                        inputNode.value = this.inputData[i + index];
                    });
                }

                // Apply activation function on non-input layer.
                layer.collection.forEach((sourceNode) => {
                    if (!input) {
                        const activation = this.activations[layer.activation];
                        sourceNode.weightedSum = sourceNode.value;
                        sourceNode.value += layer.bias;
                        sourceNode.value = activation(sourceNode.value);
                    }

                    // Start gradient descent backpropagation on last layer.
                    if (output) {
                        const error = this.costs[this.costFunction](this.targetData[i], sourceNode.value);

                        // Attempt to adjust learning rate.
                        if (this.adaptive) {
                            this.stochasticLearningRateAdaptation(this.error, error);
                        }

                        // Store the error.
                        this.error = error;
                        if (this.errorTracking) { console.log(this.error); }

                        // Backpropagate every time we're not predicting.
                        if (feedForwardOnly === false) {
                            generalBackpropagation(this.targetData[i], this.costFunction, this.learningRate, this.layers);
                        }

                        // Generate output value.
                        this.output.push(sourceNode.value);

                        // Reset all node values after iteration.
                        this.clearNodeValues();
                    }

                    // Adjust target node value (add to it's weighted sum). This is faster than dot product
                    // on non gpu accelerated devices.
                    sourceNode.connectedTo.forEach((sourceConnectionObject) => {
                        sourceConnectionObject.node.value += sourceConnectionObject.weight * sourceNode.value;
                    });
                });
            });
        }

        return this.error;
    }

    /**
     * Clear all node values (after every iteration).
     */
    public clearNodeValues() {
        // Simple iteration over all nodes.
        this.layers.forEach(layer => {
            layer.collection.forEach(node => {
                node.value = 0;
            });
        });
    }

    /**
     * Attempt to adjust learning rate based on error tendencies.
     * 
     * @param oldError - Previous error
     * @param newError - New error
     */
    private stochasticLearningRateAdaptation(oldError: number, newError: number) {
        // Error has increased.
        if (newError > oldError) {
            this.errorCounter += 1;
            this.subsequentErorReducingIterations = 0;

            // Check if number of allowed subsequent errors has passed.
            if (this.errorCounter === this.errorFluctuationLimit) {
                // Adjust learning rate++.
                if (!this.errorCorrectionIncrease) {
                    this.learningRate += randNum(
                        this.learningRateTuneRates.min,
                        this.learningRateTuneRates.max,
                        4,
                    );
                    this.errorCorrectionIncrease = true;
                    // Adjust learning rate--.
                } else {
                    this.learningRate -= randNum(
                        this.learningRateTuneRates.min,
                        this.learningRateTuneRates.max,
                        4,
                    );
                    this.errorCorrectionIncrease = false;
                }
            }
        // Error has decreased.
        } else {
            this.errorCounter = 0;
            this.subsequentErorReducingIterations += 1;
        }
    }
};

import {
    NodeGroup,
    TerminalReasons,
    ActivationFunctionsCollection,
    CostFunctionTypes,
    CostFunctionsCollection,
    Optimizer,
} from '../../common/structures/types.struct';

import {
    Getlib
} from '../../common/structures/general/decorators';

import {
    generalBackpropagation,
} from '../../common/lib/scientific.lib';

import { shuffleArray } from '../../common/lib/utils.lib';
import {
    SGD_DECAY_RATE,
    RECOMMENDED_DIVERGENCE_THRESHOLD,
    RECOMMENDED_TRAINING_CONFIDENCE,
} from '../../common/structures/constants.struct';

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
     * Collection of training errors.
     */
    public errorCollection: number[] = [];

    /**
     * Provided input data.
     */
    public inputData: number[] = [];

    /**
     * Expected data.
     */
    private targetData: number[] = [];

    /**
     * Generated descriptive T2 layer object.
     */
    private layers: NodeGroup[];

    /**
     * Chosen error function.
     */
    private costFunction: CostFunctionTypes;

    /**
     * Optimizer function.
     */
    private optimizer!: Optimizer;

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
     * Collection of activation functions.
     */
    private activations: ActivationFunctionsCollection;

    /**
     * Collection of cost functions.
     */
    private costs: CostFunctionsCollection;

    /**
     * Store layers reference locally and modify it on the fly.
     * 
     * @param layers - T2 layers object
     * @param costFunction - Error function to be used
     * @param optimizer - Optimizer function to be used
     * @param batchSize - Training sample size
     * @param learningRate - The learning rate
     * @param errorTracking - Whether to track errors during training
     */
    constructor(
        layers: NodeGroup[],
        costFunction: CostFunctionTypes,
        optimizer: Optimizer,
        batchSize: number,
        learningRate: number = 1,
        errorTracking: boolean = false
    ) {
        this.layers = layers;
        this.costFunction = costFunction;
        this.optimizer = optimizer;
        this.batchSize = batchSize;
        this.learningRate = learningRate;
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
            && this.error >= RECOMMENDED_DIVERGENCE_THRESHOLD
        ) {
            return 'Possible divergence detected.';
        } else if (
            reasons.includes('convergence')
            && this.error <= RECOMMENDED_TRAINING_CONFIDENCE
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
        shuffleArray(this.inputData);

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

                        // Store the error.
                        if (!feedForwardOnly) {
                            this.errorCollection.push(error);
                            this.error = this.errorCollection.reduce((a, b) => a + b) / this.errorCollection.length;
                        }
                        if (this.errorTracking) { console.log(this.error); }

                        // Backpropagate every time we're not predicting.
                        if (feedForwardOnly === false && this.optimizer === 'SGD') {
                            generalBackpropagation(
                                this.targetData[i],
                                this.costFunction,
                                this.learningRate, this.layers,
                                (gradient, learningRate, prevDelta) => (gradient * learningRate) + prevDelta * SGD_DECAY_RATE,
                            );
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
};

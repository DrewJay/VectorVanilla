import {
    NodeGroup,
    ActivationFunctionsCollection,
    CostFunctionTypes,
    CostFunctionsCollection,
    DerivativeFunctionsCollection,
} from '../../common/structures/types.struct';

import {
    Getlib
} from '../../common/structures/general/decorators';

import {
    deltaRule,
} from '../../common/lib/scientific.lib';

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
    private layers: NodeGroup[];

    /**
     * Chosen error function.
     */
    private costFunction: CostFunctionTypes;

    /**
     * Learning rate coefficient.
     */
    private learningRate: number = 1;

    /**
     * Deremine whether to track errors during iterations.
     */
    private errorTracking = false;

    /**
     * Collection of activation functions.
     */
    public activations: ActivationFunctionsCollection;

    /**
     * Collection of cost functions.
     */
    public costs: CostFunctionsCollection;

    /**
     * Collection of derivatives of activation functions.
     */
    public derivatives: DerivativeFunctionsCollection;

    /**
     * Store layers reference locally and modify it on the fly.
     * 
     * @param layers - T2 layers object
     * @param costFunction - Error function to be used
     * @param learningRate - The learning rate (min. 1)
     * @param errorTracking - Whether to track errors during training
     */
    constructor(
        layers: NodeGroup[],
        costFunction: CostFunctionTypes,
        learningRate: number = 1,
        errorTracking: boolean = false
    ) {
        this.layers = layers;
        this.costFunction = costFunction;
        this.learningRate = learningRate;
        this.errorTracking = errorTracking;
    }

    /**
     * Initialize data that will enter the network.
     * 
     * @param data - input dataset
     */
    public initializeInputData(data: number[]) {
        this.inputData = data;
    }

    /**
     * Initialize data that will enter the network.
     * 
     * @param data - input dataset
     */
    public initializeTargetData(data: number[]) {
        this.targetData = data;
    }

    /**
     * Distribute data across the neural network.
     */
    public async iterate() {
        for (let i = 0; i < this.inputData.length; i++) {
            this.layers.forEach((layer) => {
                // Load first layer with input data.
                const input = layer.flags.includes('input');
                const output = layer.flags.includes('output');

                if (input) {
                    layer.collection[0].value = this.inputData[i];
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
                        layer.error = this.costs[this.costFunction](this.targetData[i], sourceNode.value);
                        if (this.errorTracking) { console.log(layer.error); }
                        this.backpropagate(this.targetData[i]);
                    }

                    // Adjust target node value (add to it's weighted sum).
                    sourceNode.connectedTo.forEach((sourceConnectionObject) => {
                        sourceConnectionObject.node.value += sourceConnectionObject.weight * sourceNode.value;
                    });
                });
            });
        }
    }

    /**
     * Backpropagate and adjusts node weights.
     * 
     * @param target - Expected output number
     */
    private backpropagate(target: number) {
        // Reverse layer iteration.
        for (let i = this.layers.length - 1; i > -1; i--) {
            const layer = this.layers[i];

            layer.collection.forEach((sourceNode) => {
                const id = sourceNode.id;

                // Apply delta-rule to adjust neural network weights.
                sourceNode.connectedBy.forEach((sourceConnectionObject) => {
                    let delta = deltaRule(
                        target,
                        sourceNode.value,
                        this.derivatives[layer.activation](sourceNode.weightedSum),
                        sourceConnectionObject.node.value,
                        this.learningRate
                    );
                    
                    sourceConnectionObject.weight -= delta;

                    // Propagate delta change to connected node.
                    const targetConnectionObject = sourceConnectionObject.node.connectedTo.find((targetConnectionObject) => targetConnectionObject.node.id === id);
                    targetConnectionObject.weight = sourceConnectionObject.weight;
                });
            });
        }
    }
};

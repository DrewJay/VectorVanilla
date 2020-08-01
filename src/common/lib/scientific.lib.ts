import {
    RRGenerator,
    Connection,
    NodeGroup,
    CostFunctionTypes,
} from '../structures/types.struct';
import {
    randNum,
    zeros,
    mean,
} from './utils.lib';
import {
    crossMult,
    scalarMult,
    crossAdd,
} from './matrix.lib';

// Derivative functions.
const derivatives = require('./functions/derivatives.lib');

/**
 * Xavier normal weight initialization formula.
 * 
 * For more info see:
 * http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?
 * 
 * @param indim - Input dimensions
 * @param outdim - Output dimensions
 * @param specialized - Special case coefficient multiplication
 * @param absolutize - Only positive values are allowed
 * @returns Initial weight value.
 */
export const XavierNormal: RRGenerator = (
    indim: number,
    outdim: number,
    specialized = false,
    absolutize = false,
) => {
    const upscale = specialized ? 4 : 1;
    const bound = Math.sqrt(6 / (indim + outdim)) * upscale;
    const result = randNum(-bound, bound, 2);
    return absolutize ? Math.abs(result) : result;
}

/**
 * Simple perceptron backpropagation rule.
 * 
 * For more info see:
 * https://en.wikipedia.org/wiki/Delta_rule
 * https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 * 
 * @param targetValue - Expected value
 * @param sourceNodeValue - Current node value
 * @param weightedSumDerivative - Derivative of activation applied on weighted sum
 * @param inputValue - incomming value
 * @param learningRate - Learning rate coefficient
 */
export const deltaRule = (
    targetValue: number,
    sourceNodeValue: number,
    weightedSumDerivative: number,
    inputValue: number,
    learningRate: number,
) => ((targetValue - sourceNodeValue) * weightedSumDerivative * inputValue * learningRate);

/**
 * Generalized delta rule, for n-layer neural network.
 *
 * For more info see:
 * https://en.wikipedia.org/wiki/Backpropagation
 * 
 * @param target - Target value
 * @param costFunction - Currently used cost function
 * @param learningRate - Used when calculating delta weight
 * @param layers - Object describing network layers
 */
export const generalBackpropagation = (
    target: number[],
    costFunction: CostFunctionTypes,
    learningRate: number,
    layers: NodeGroup[],
    optimizer: (gradient: number[], learningRate: number, prevDelta: number) => number[],
) => {
    // Reverse layer iteration.
    for (let i = layers.length - 1; i > -1; i--) {
        const layer = layers[i];

        layer.collection.forEach((sourceNode) => {
            // Output layer.
            if (i === layers.length - 1) {
                // Sigma is important component in backpropagation. It is virtually
                // calculated as a multiplication of derivatives of cost function
                // and activation funcion.
                const sigma = mean(scalarMult(
                    derivatives[layer.activation](sourceNode.weightedSum) as number[],
                    derivatives[costFunction](target, sourceNode.value) as number,
                ));

                sourceNode.sigma = sigma;
                // Iterate over neurons connected to output neuron.
                sourceNode.connectedBy.forEach((sourceConnectionObject) => {
                    // Calculate final delta weight. It equals sigma times source node value.
                    const gradient = scalarMult(sourceConnectionObject.node.value, sigma);

                    const delta = mean(optimizer(gradient, learningRate, sourceConnectionObject.prevDelta));
                    sourceConnectionObject.weight -= delta;
                    sourceConnectionObject.prevDelta = delta;

                    // Get to the other side of the connection and propagate delta weight over there.
                    const targetConnectionObject = sourceConnectionObject.node.connectedTo.find((target) => target.node.id === sourceNode.id);
                    targetConnectionObject.weight = sourceConnectionObject.weight;
                });
            // Hidden layer neurons.
            } else {
                // Get sigma sum from next layer.
                const sum = sigmaSum(sourceNode.connectedTo, target.length);
                sourceNode.sigma = mean(scalarMult(derivatives[layer.activation](sourceNode.value) as number[], sum));

                // Apply delta-rule to adjust neural network weights.
                sourceNode.connectedBy.forEach((sourceConnectionObject) => {
                    // Calculate and apply delta weight.
                    const gradient = scalarMult(sourceConnectionObject.node.value, sourceNode.sigma);

                    const delta = mean(optimizer(gradient, learningRate, sourceConnectionObject.prevDelta));
                    sourceConnectionObject.weight -= delta;
                    sourceConnectionObject.prevDelta = delta;

                    // Get to the other side of the connection and propagate delta weight over there.
                    const targetConnectionObject = sourceConnectionObject.node.connectedTo.find((target) => target.node.id === sourceNode.id);
                    targetConnectionObject.weight = sourceConnectionObject.weight;
                });        
            }
        });
    }
};

/**
 * Calculate sigma sum used in backpropagation.
 *
 * @param nextLayerConnections - Connection collection of layer to the right
 * @param targetLength - Used to initialize sum variable
 * @returns Summed sigma values
 */
export const sigmaSum = (nextLayerConnections: Connection[], targetLength: number) => {
    let sum = 0;
    nextLayerConnections.forEach((sourceConnectionObject) => {
        const increment = sourceConnectionObject.node.sigma * sourceConnectionObject.weight;
        sum += increment;
    });

    return sum;
};

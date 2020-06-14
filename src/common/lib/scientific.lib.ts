import { RRGenerator } from '../structures/types.struct';
import { randNum } from './utils.lib';

/**
 * Xavier normal weight initialization formula.
 * 
 * For more info see:
 * http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?
 * 
 * @param indim - Input dimensions
 * @param outdim - Output dimensions
 * @param specialized - Special case coefficient multiplication
 * @returns Initial weight value.
 */
export const XavierNormal: RRGenerator = (indim: number, outdim: number, specialized = false) => {
    const upscale = specialized ? 4 : 1;
    const bound = Math.sqrt(6 / (indim + outdim)) * upscale;
    return randNum(-bound, bound, 2);
};

/**
 * Complicated stuff also known as dela rule.
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
) => (targetValue - sourceNodeValue) * weightedSumDerivative * inputValue * learningRate;

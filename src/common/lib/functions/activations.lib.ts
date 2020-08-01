import {
    ActivationFunctionsCollection,
} from '../../structures/types.struct';

module.exports = {
    /**
     * Apply sigmoid function on a value.
     *
     * @param input - The input number
     * @returns Sigmoid'd value
     */
    sigmoid: (input: number[]) => {
        return input.map((value) => 1 / (1 + Math.exp(-value)));
    },

    /**
     * Apply ReLu function on a value.
     * 
     * @param input - The input number
     * @returns ReLu'd value
     */
    ReLu: (input: number[]) => {
        return input.map((value) => ((value < 0) ? 0 : value));
    },

    /**
     * Sine activation function.
     * 
     * @param input - The input number
     * @returns Sine value
     */
    sin: (input: number[]) => {
        return input.map((value) =>  Math.sin(value));
    },

    /**
     * Cosine activation function.
     * 
     * @param input - The input number
     * @returns Sine value
     */
    cos: (input: number[]) => {
        return input.map((value) => Math.cos(value));
    },

    /**
     * Tangent activation function.
     * 
     * @param input - The input number
     * @returns Tangent value
     */
    tan: (input: number[]) => {
        return input.map((value) => Math.tan(value));
    },

    /**
     * Hyperbolic tangent activation function.
     * 
     * @param input - The input number
     * @returns Hyperbolic tangent value
     */
    tanh: (input: number[]) => {
        return input.map((value) => Math.tanh(value));
    },

    /**
     * Natural log activation function.
     * 
     * @param input - The input number
     * @returns Natlog value
     */
    log: (input: number[]) => {
        return input.map((value) => Math.log(value));
    },

    /**
     * Value passer.
     * 
     * @param input - The input number
     * @returns Input value
     */
    none: (input: number[]) => {
        return input.map((value) => value);
    },

    /**
     * Apply inear function on a value.
     * 
     * @param slope - Function slope parameter
     * @param input - The x coordinate
     * @returns Linear'd value
     */
    linear: (slope: number[] = [1], input: number[] = [1]) => {
        return slope.map((value) => value * input[0]);
    },
} as ActivationFunctionsCollection;

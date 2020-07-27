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
    sigmoid: (input: number) => {
        return 1 / (1 + Math.exp(-input));
    },

    /**
     * Apply ReLu function on a value.
     * 
     * @param input - The input number
     * @returns ReLu'd value
     */
    ReLu: (input: number) => {
        return ((input < 0) ? 0 : input);
    },

    /**
     * Sine activation function.
     * 
     * @param input - The input number
     * @returns Sine value
     */
    sin: (input: number) => {
        return Math.sin(input);
    },

    /**
     * Cosine activation function.
     * 
     * @param input - The input number
     * @returns Sine value
     */
    cos: (input: number) => {
        return Math.cos(input);
    },

    /**
     * Tangent activation function.
     * 
     * @param input - The input number
     * @returns Tangent value
     */
    tan: (input: number) => {
        return Math.tan(input);
    },

    /**
     * Hyperbolic tangent activation function.
     * 
     * @param input - The input number
     * @returns Hyperbolic tangent value
     */
    tanh: (input: number) => {
        return Math.tanh(input);
    },

    /**
     * Natural log activation function.
     * 
     * @param input - The input number
     * @returns Natlog value
     */
    log: (input: number) => {
        return Math.log(input);
    },

    /**
     * Value passer.
     * 
     * @param input - The input number
     * @returns Input value
     */
    none: (input: number) => {
        return input;
    },

    /**
     * Apply inear function on a value.
     * 
     * @param slope - Function slope parameter
     * @param input - The x coordinate
     * @returns Linear'd value
     */
    linear: (slope: number = 1, input: number = 1) => {
        return slope * input;
    },
} as ActivationFunctionsCollection;

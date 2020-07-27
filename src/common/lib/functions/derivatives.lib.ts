import {
    DerivativeFunctionsCollection,
} from '../../structures/types.struct';

module.exports = {
    /**
     * Sigmoid derivative function.
     *
     * @param input - The input number
     * @returns Sigmoid derivative value
     */
    sigmoid: (input: number) => {
        const _sigmoid = 1 / (1 + Math.exp(-input));
        return _sigmoid * (1 - _sigmoid);
    },

    /**
     * ReLu derivative function
     * 
     * @param input - The input number
     * @returns ReLu derivative value
     */
    ReLu: (input: number) => {
        return (input >= 0) ? 1 : 0;
    },


    /**
     * Sine derivative function.
     * 
     * @param input - The input number
     * @returns Sine derivative value
     */
    sin: (input: number) => {
        return Math.cos(input);
    },

    /**
     * Cosine derivative function.
     * 
     * @param input - The input number
     * @returns Cosine derivative value
     */
    cos: (input: number) => {
        return -Math.sin(input);
    },

    /**
     * Tangent derivative function.
     * 
     * @param input - The input number
     * @returns Tangent derivative value
     */
    tan: (input: number) => {
        return Math.pow(1 / Math.cos(input), 2);
    },

    /**
     * Hyperbolic tangent derivative function.
     * 
     * @param input - The input number
     * @returns Hyperbolic tangent derivative value
     */
    tanh: (input: number) => {
        return 1 / Math.pow(Math.cosh(input), 2);
    },

    /**
     * Natural log derivative function.
     * 
     * @param input - The input number
     * @returns Natlog derivative value
     */
    log: (input: number) => {
        return (input === 0) ? 0 : (1 / input);
    },

    /**
     * Value passer derivativee.
     * 
     * @returns 1 value
     */
    none: () => {
        return 1;
    },

    /**
     * Linear function derivative.
     * 
     * @param slope - Function slope parameter
     * @returns Linear derivative value
     */
    linear: (slope: number = 1) => {
        return slope; 
    },

    /**
     * Mean squared error derivative.
     *
     * @param target - Expected number
     * @param output - Gotten number
     * @returns Derivated MSE value
     */
    meanSquaredError: (target: number, output: number) => {
        return output - target;
    }
} as DerivativeFunctionsCollection;

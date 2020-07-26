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

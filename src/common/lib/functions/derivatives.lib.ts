
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
     * @param input - The input number
     * @param slope - Function slope parameter
     * @returns Linear derivative value
     */
    linear: (input: number, slope: number) => {
        if (input === 0) { return 0; }
        return slope * (input / input);
    },
} as DerivativeFunctionsCollection;

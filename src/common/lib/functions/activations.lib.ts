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

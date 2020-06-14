
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
     * @param input - The input number
     * @param slope - Function slope parameter
     * @returns Linear'd value
     */
    linear: (input: number, slope: number) => {
        return slope * input;
    },
} as ActivationFunctionsCollection;

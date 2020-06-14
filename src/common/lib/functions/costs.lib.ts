
import {
    CostFunctionsCollection,
} from '../../structures/types.struct';

module.exports = {
    /**
     * Calculate mean squared error.
     *
     * @param target - Expected number
     * @param output - Gotten number
     * @returns Mean squared error value
     */
    meanSquaredError: (target: number, output: number) => {
        return .5 * (target - output) ** 2;
    },
} as CostFunctionsCollection;

import {
    CostFunctionsCollection,
} from '../../structures/types.struct';

import {
    mean,
} from '../../lib/utils.lib';

module.exports = {
    /**
     * Calculate mean squared error.
     *
     * @param target - Expected number
     * @param output - Gotten number
     * @returns Mean squared error value
     */
    meanSquaredError: (target: number[], output: number[]) => {
        const meanTarget = mean(target);
        const meanOutput = mean(output);
        return .5 * (meanTarget - meanOutput) ** 2;
    },
} as CostFunctionsCollection;

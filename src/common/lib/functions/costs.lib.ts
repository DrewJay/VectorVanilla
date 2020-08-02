import {
    CostFunctionsCollection,
} from '../../structures/types.struct';

import {
    mean,
} from '../../lib/utils.lib';

import {
    crossSub,
} from '../../lib/matrix.lib';

module.exports = {
    /**
     * Calculate mean squared error.
     *
     * @param target - Expected number
     * @param output - Gotten number
     * @returns Mean squared error value
     */
    meanSquaredError: (target: number[], output: number[]) => {
        const sub = crossSub(target, output);
        const preform = sub.map((val) => .5 * val ** 2);
        return mean(preform);
    },
} as CostFunctionsCollection;

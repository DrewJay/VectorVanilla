import { DistributionUnit } from '../operators/distribution.operator';
import { plot, Plot } from 'nodeplotlib';

/**
 * Class collecting final information from operator classes.
 */
export class ResultCollector {
    /**
     * Generate prediction output from input data (generic).
     *
     * @param distributionUnit - Distribution unit class instance
     * @param input - Neural network input data
     * @returns Prediction output.
     */
    static predict(distributionUnit: DistributionUnit, input: number[]) {
        distributionUnit.inputData = input;
        distributionUnit.iterate(true);
        return distributionUnit.output;
    }

    /**
     * Plot predictions.
     *
     * @param features - Feature data array
     * @param labels - Label data array
     */
    static plot(features: number[], labels: number[]) {
        const plotdata: Plot[] = [{x: features, y: labels, type: 'scatter'}];
        plot(plotdata);
    }
};

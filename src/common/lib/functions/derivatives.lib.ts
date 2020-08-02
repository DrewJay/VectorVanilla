import {
    DerivativeFunctionsCollection,
} from '../../structures/types.struct';

import {
    crossMult,
    scalarAdd,
    crossSub,
} from '../matrix.lib';

import {
    mean,
} from '../utils.lib';

module.exports = {
    /**
     * Sigmoid derivative function.
     *
     * @param input - The input number
     * @returns Sigmoid derivative value
     */
    sigmoid: (input: number[]) => {
        const _sigmoid = input.map((value) => 1 / (1 + Math.exp(-value)));
        return crossMult(_sigmoid, scalarAdd(_sigmoid, -1));
    },

    /**
     * ReLu derivative function
     * 
     * @param input - The input number
     * @returns ReLu derivative value
     */
    ReLu: (input: number[]) => {
        return input.map((value) => (value >= 0) ? 1 : 0);
    },


    /**
     * Sine derivative function.
     * 
     * @param input - The input number
     * @returns Sine derivative value
     */
    sin: (input: number[]) => {
        return input.map((value) => Math.cos(value));
    },

    /**
     * Cosine derivative function.
     * 
     * @param input - The input number
     * @returns Cosine derivative value
     */
    cos: (input: number[]) => {
        return input.map((value) => -Math.sin(value));
    },

    /**
     * Tangent derivative function.
     * 
     * @param input - The input number
     * @returns Tangent derivative value
     */
    tan: (input: number[]) => {
        return input.map((value) => Math.pow(1 / Math.cos(value), 2));
    },

    /**
     * Hyperbolic tangent derivative function.
     * 
     * @param input - The input number
     * @returns Hyperbolic tangent derivative value
     */
    tanh: (input: number[]) => {
        return input.map((value) => 1 / Math.pow(Math.cosh(value), 2));
    },

    /**
     * Natural log derivative function.
     * 
     * @param input - The input number
     * @returns Natlog derivative value
     */
    log: (input: number[]) => {
        return input.map((value) => (value === 0) ? 0 : (1 / value));
    },

    /**
     * Value passer derivative.
     * 
     * @returns 1 value
     */
    none: (input: number[]) => {
        return input.map(() => 1);
    },

    /**
     * Linear function derivative.
     * 
     * @param slope - Function slope parameter
     * @returns Linear derivative value
     */
    linear: (slope: number[] = [1]) => {
        return slope;
    },

    /**
     * Mean squared error derivative.
     *
     * @param target - Expected number
     * @param output - Gotten number
     * @returns Derivated MSE value
     */
    meanSquaredError: (target: number[], output: number[]) => {
        return mean(crossSub(output, target));
    }
} as DerivativeFunctionsCollection;

/**
 * Generate random number in range with specific precision.
 * 
 * @param low - Lower bound
 * @param high - Upper bound
 * @param float - Number of floating points
 * @returns Given number
 */
export const randNum = (low: number, high: number, float: number): number => {
    return parseFloat((Math.random() * (low - high) + high).toFixed(float));
};

/**
 * Generate random string of any length.
 *
 * @param length - String length.
 * @returns Random string
 */
export const randStr = (length: number) => {
    var randomstring = require('randomstring') as any;
    return randomstring.generate(length);
};

/**
 * Calculate dot product from two arrays.
 * 
 * @param arr1 - Feature array 1
 * @param arr2 - Feature array 2
 * @returns Final dot product
 */
export const dotProduct = (arr1: number[], arr2: number[]) => {
    return arr1.map((value, index) => value * arr2[index]).reduce((value, next) => value + next);
};

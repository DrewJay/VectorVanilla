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

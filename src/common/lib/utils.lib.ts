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
 * Get mean value of an array.
 *
 * @param arr - Input array
 * @returns Array mean value
 */
export const mean = (arr: number[]) => {
    return arr.reduce((a, b) => a + b) / arr.length;
};

/**
 * Generate row array filled with zeros.
 *
 * @param length - Amount of zeros
 * @returns Particular array 
 */
export const zeros = (length: number) => {
    return new Array(length).fill(0);
};

/**
 * Fisher-Yates array shuffling method.
 *
 * @param array - Input array
 * @returns Shuffled array
 */
export const shuffleArray = (arrayCollection: [number[][], number[][]]) => {
    let primary = arrayCollection[0];
    for (let i = primary.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        for (let q = 0; q < arrayCollection.length; q++) {
            const array = arrayCollection[q];
            var temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
};

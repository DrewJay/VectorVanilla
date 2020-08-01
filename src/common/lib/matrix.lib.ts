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

/**
 * Flatten an array.
 *
 * @param arr - Input array
 * @returns Flattened array
 */
export const flatten = (arr: number[][]) => {
    return arr.reduce((acc, val) => acc.concat(val), []);
};

/**
 * Multiply array by scalar value.
 * 
 * @param arr - Input array
 * @param scalar - Scalar value
 * @returns Operation result
 */
export const scalarMult = (arr: number[], scalar: number) => {
    return arr.map((value) => value * scalar);
};

/**
 * Add scalar to array value.
 * 
 * @param arr - Input array
 * @param scalar - Scalar value
 * @returns Operation result
 */
export const scalarAdd = (arr: number[], scalar: number) => {
    return arr.map((value) => value + scalar);
};

/**
 * Cross-array multiplication.
 *
 * @param arr1 - Source array
 * @param arr2 - Target array
 * @returns Operation result
 */
export const crossMult = (arr1: number[], arr2: number[]) => {
    return arr1.map((value, index) => value * arr2[index]);  
};

/**
 * Cross-array addition.
 *
 * @param arr1 - Source array
 * @param arr2 - Target array
 * @returns Operation result
 */
export const crossAdd = (arr1: number[], arr2: number[]) => {
    return arr1.map((value, index) => value + arr2[index]);  
};
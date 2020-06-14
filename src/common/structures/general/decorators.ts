/**
 * Dynamically load whatever data when initializing a class.
 *
 * @param libdata - Object of member variable names and paths
 * @returns New supreme constructor
 */
export const Getlib = (libdata: { [key: string]: string }) => (constructor: Function): any => {
    for (let key in libdata) {
        constructor.prototype[key] = require.main.require(`./${libdata[key]}`);
    }
    return constructor;
};
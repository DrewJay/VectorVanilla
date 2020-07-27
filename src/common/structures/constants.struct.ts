/**
 * SGD alpha parameter (decay)
 */
export const SGD_DECAY_RATE = 0.9;

/**
 * If error reaches this value, we can stop iterating in particular circumstances.
 */
export const RECOMMENDED_DIVERGENCE_THRESHOLD = 2.0;

/**
 * If error has this value, we can possibly stop iterating.
 */
export const RECOMMENDED_TRAINING_CONFIDENCE = Math.pow(10, -30);
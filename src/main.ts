import { NetworkAbstractionUnit } from './core/bundlers/abstraction.bundler';
import { DistributionUnit } from './core/operators/distribution.operator';

// Use this class to create descriptive NN model.
const net = new NetworkAbstractionUnit();

// Describe network layers.
net.add({ nodes: 1, activation: 'sigmoid', bias: 0, });
net.add({ nodes: 10, activation: 'sigmoid', bias: 0, });
net.add({ nodes: 10, activation: 'sigmoid', bias: 0, });
net.add({ nodes: 1, activation: 'sigmoid', bias: 0, });

// Use NetworkAbastractionUnit API to forge detailed network model.
net.describeLayers();
net.formConnections();
net.initializeWeights();

// Prepare model for data distributiond and adjustments using DistributionUnit class.
const dist = new DistributionUnit(
    net.layerStackT2,
    'meanSquaredError',
    .4,
    true,
);

// Prepare simple datasets.
dist.initializeInputData([1, 2, 3, 4, 5]);
dist.initializeTargetData([2, 4, 6, 8, 10]);

// Use DistributionUnit iterative generator method to perform iterations.
for (let i = 0; i < 300; i++) {
    dist.iterate().next();
}
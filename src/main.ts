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
    1,
    .0001,
    false,
);

// Prepare simple datasets.
const data = [];
for (let i = 0; i < 120; i++) { data[i] = i/1000; }
const plusdata = data.map((val) => val + .0001);

dist.initializeInputData(data);
dist.initializeTargetData(plusdata);

// Use DistributionUnit iterative generator method to perform iterations.
for (let i = 0; i < 500; i++) {
    dist.iterate();
}

// Show final error.
console.log(net.layerStackT2[net.layerStackT2.length - 1].error);

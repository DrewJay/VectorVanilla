import { NetworkAbstractionUnit } from './core/bundlers/abstraction.bundler';
import { DistributionUnit } from './core/operators/distribution.operator';
import { ResultCollector } from './core/collectors/result.collector';

// Use this class to create descriptive NN model.
const net = new NetworkAbstractionUnit();

// Describe network layers.
net.add({ nodes: 1, activation: 'none', bias: 0, });
net.add({ nodes: 20, activation: 'ReLu', bias: 0, });
net.add({ nodes: 20, activation: 'ReLu', bias: 0, });
net.add({ nodes: 1, activation: 'sin', bias: 0, });

// Use NetworkAbastractionUnit API to forge detailed network model.
net.describeLayers();
net.formConnections();
net.initializeWeights();

// Prepare model for data distribution and adjustments using DistributionUnit class.
const dist = new DistributionUnit(
    net.layerStackT2,
    'meanSquaredError',
    'SGD',
    1,
    .1,
    false,
);

// Prepare simple datasets.
const data = [];
let increment = 0;

for (let i = 0; i < 6; i += .01) { data[increment++] = i / 10; }
const targetData = data.map((val) => Math.sin(val));

// Plot target data.
const _data = [...data];
const _targetData = [...targetData];
ResultCollector.plot(_data, _targetData);

dist.initializeInputData(data);
dist.initializeTargetData(targetData);

// Use DistributionUnit iterative generator method to perform iterations.
for (let i = 0; i < 100; i++) {
    dist.epoch(false, true);
}

// Show final error.
console.log(`Error = ${dist.getError()}.`);

// Reset distribution instance.
dist.reset();

// Make prediction.
const results = { x: [], y: [], };

// Generate prediction input data.
for (let i = 0; i < 6; i += 0.01) {
    const feature = i;
    const prediction = ResultCollector.predict(dist, [feature]);
    results.x.push(feature);
    results.y.push(prediction[0]);
}

// Plot predicted data.
ResultCollector.plot(results.x, results.y);

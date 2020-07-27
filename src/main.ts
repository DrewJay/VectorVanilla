import { NetworkAbstractionUnit } from './core/bundlers/abstraction.bundler';
import { DistributionUnit } from './core/operators/distribution.operator';
import { ResultCollector } from './core/collectors/result.collector';

// Use this class to create descriptive NN model.
const net = new NetworkAbstractionUnit();

// Describe network layers.
net.add({ nodes: 1, activation: 'none', bias: 0, });
net.add({ nodes: 100, activation: 'tanh', bias: 0, });
net.add({ nodes: 100, activation: 'ReLu', bias: 0, });
net.add({ nodes: 100, activation: 'ReLu', bias: 0, });
net.add({ nodes: 1, activation: 'linear', bias: 0, });

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
    .0009,
    false,
);

// Prepare simple datasets.
const data = [];
let increment = 0;

for (let i = 1; i < 4; i += .01) { data[increment++] = i / 10; }
const targetData = data.map((val) => Math.sin(val) / 10);
const errors = [];
const iterations = [];

// Plot target data.
ResultCollector.plot(data, targetData);

dist.initializeInputData(data);
dist.initializeTargetData(targetData);

// Use DistributionUnit iterative generator method to perform iterations.
for (let i = 0; i < 500; i++) {
    const err = dist.iterate();
    errors.push(err);
    iterations.push(i);
    const reason = dist.terminateOn(['divergence']);
    if (reason) { console.log(reason); break; }
}

// Show final error.
console.log(`Error = ${dist.error}.`);

// Make prediction.
const results = { x: [], y: [], };

// Generate prediction input data.
for (let i = 1; i < 4; i += 0.01) {
    const feature = i;
    const prediction = ResultCollector.predict(dist, [feature]);
    results.x.push(feature);
    results.y.push(prediction[0]);
}

// Plot predicted data.
ResultCollector.plot(results.x, results.y);

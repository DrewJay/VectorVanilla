import { NetworkAbstractionUnit } from './core/bundlers/abstraction.bundler';
import { DistributionUnit } from './core/operators/distribution.operator';
import { ResultCollector } from './core/collectors/result.collector';

// Use this class to create descriptive NN model.
const net = new NetworkAbstractionUnit();

// Describe network layers.
net.add({ nodes: 1, activation: 'ReLu', bias: 0, });
net.add({ nodes: 5, activation: 'ReLu', bias: 0, });
net.add({ nodes: 1, activation: 'linear', bias: 0, });

// Use NetworkAbastractionUnit API to forge detailed network model.
net.describeLayers();
net.formConnections();
net.initializeWeights();

// Prepare model for data distribution and adjustments using DistributionUnit class.
const dist = new DistributionUnit(
    net.layerStackT2,
    'meanSquaredError',
    1,
    1,
    true,
    false,
);

// Prepare simple datasets.
const data = [];
for (let i = 1; i < 2000; i += 1) { data[i - 1] = i/10000; }
const targetData = data.map((val) => val * 7);

dist.initializeInputData(data);
dist.initializeTargetData(targetData);

// Use DistributionUnit iterative generator method to perform iterations.
for (let i = 0; i < 1000; i++) {
    const err = dist.iterate();
    const reason = dist.terminateOn(['divergence']);
    if (reason) { console.log(reason); break; }
}

// Show final error.
console.log(`Error = ${dist.error}.`);

// Make prediction.
const results = { x: [], y: [], };

// Generate prediction input data.
for (let i = 0; i < 100; i ++) {
    const feature = i/100000;
    const prediction = ResultCollector.predict(dist, [feature]);
    results.x.push(feature * i * 100);
    results.y.push(prediction[0] * i * 100);
}

// Plot the results.
ResultCollector.plot(results.x, results.y);

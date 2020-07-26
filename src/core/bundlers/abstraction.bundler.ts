import {
    Layer,
    NodeGroup,
} from '../../common/structures/types.struct';
import { XavierNormal } from '../../common/lib/scientific.lib';
import { randStr } from '../../common/lib/utils.lib';


/**
 * This method generates network abstraction structures from very
 * basic numeric/textual inputs.
 */
export class NetworkAbstractionUnit {
    /**
     * Tier 1 layer stack (lower abstraction).
     */
    public layerStackT1: Layer[] = [];

    /**
     * Tier 2 layer stack (high abstraction).
     */
    public layerStackT2: NodeGroup[] = []; 

    /**
     * Simply add requested layer to layer stack.
     *
     * @param layer - Layer to add to stack.
     */
    public add(layer: Layer) {
        this.layerStackT1.push(layer);
    }

    /**
     * First transformation step - take layer stack
     * and create deeper description using Node objects.
     */
    public describeLayers() {
        this.layerStackT1.forEach((layer, index) => {
            this.layerStackT2[index] = {
                collection: [],
                activation: layer.activation,
                bias: layer.bias,
                flags: [],
            };
            
            // Generate blank node objects with random names.
            for(let i = 0; i < layer.nodes; i++) {
                this.layerStackT2[index].collection.push(
                    {
                        id: randStr(5),
                        value: 0,
                        weightedSum: 0,
                        connectedTo: [],
                        connectedBy: [],
                    },
                );
            }
        });
    }

    /**
     * Form connection references between layer's nodes.
     */
    public formConnections() {
        this.layerStackT2.forEach((nodeGroup, index) => {
            if (index === 0) {
                nodeGroup.flags.push('input');
            } else if (index === this.layerStackT2.length - 1) {
                nodeGroup.flags.push('output');
            }

            // Generate T2 layers by creating blank description objects and basic connection relations.
            if (this.layerStackT2[index + 1]) {
                nodeGroup.collection.forEach((sourceNode) => {
                    this.layerStackT2[index + 1].collection.forEach((targetNode) => {
                        sourceNode.connectedTo.push({ node: targetNode, weight: null, });
                        targetNode.connectedBy.push({ node: sourceNode, weight: null, });
                    });
                });
            }
        });
    }

    /**
     * Initialize weights for T2 layer collection.
     */
    public initializeWeights() {
        for (let i = 0; i < this.layerStackT2.length; i++) {
            // Precheck if calculations are worth making.
            if (this.layerStackT2[i + 1]) {
                const collection = this.layerStackT2[i].collection;

                // Iterate over collection nodes.
                collection.forEach((sourceNode) => {
                    const id = sourceNode.id;

                    // Distribute weights to particular connected nodes.
                    sourceNode.connectedTo.forEach((sourceConnectionObject) => {
                        const weight = XavierNormal(sourceNode.connectedBy.length, sourceNode.connectedTo.length, false, true);
                        sourceConnectionObject.weight = weight;

                        const targetConnectionObject = sourceConnectionObject.node.connectedBy.find((targetConnectionObject) => targetConnectionObject.node.id === id);
                        targetConnectionObject.weight = weight;
                    });
                });
            }
        }
    }
};

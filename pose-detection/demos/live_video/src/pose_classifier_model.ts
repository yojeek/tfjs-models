import * as tf from '@tensorflow/tfjs';

/**
 * @param inputSize
 * @param outputSize 
 * @returns 
 */
function createModel(inputSize: number, outputSize: number) {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [inputSize],
        units: inputSize,
        activation: 'relu'
    }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    const middleDenseLayerSize = Math.floor(inputSize / 2);
    model.add(
        tf.layers.dense({
            units: middleDenseLayerSize,
            activation: 'relu'
        })
    );
    model.add(tf.layers.dropout({ rate: 0.5 }));
    const labelsNumber = outputSize;
    model.add(
        tf.layers.dense({
            units: labelsNumber,
            activation: 'softmax'
        })
    );
    model.compile({
        optimizer: tf.train.adam(0.0001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

export default createModel;
/** Example model given by GTP-4 */
import * as tf from '@tensorflow/tfjs';


function createModel(inputSize: number, outputSize: number) {
    const model = tf.sequential();

    model.add(tf.layers.dense({ units: 64, activation: 'relu', inputShape: [inputSize] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: outputSize, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    })

    return model;
}

export default createModel;
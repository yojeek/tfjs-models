import * as tf from '@tensorflow/tfjs';

import { Keypoint } from '@tensorflow-models/pose-detection/types'
import { getTorsoCenter, scaleToBoundingBox } from './pose-utils';
import createModel from './pose_classifier_model';
import { log } from 'console';

export type PoseData = Keypoint[];

export interface PosesData {
    [label: string]: PoseData[];
}

/**
 * Converts an array of Keypoint objects to a TensorFlow.js tensor.
 *
 * @param {Array<Keypoint>} keypoints - An array of Keypoint objects.
 * @returns {tf.Tensor} A TensorFlow.js tensor with rank 1.
 */
function keypointToTensor(keypoints: Array<Keypoint>): tf.Tensor {
    // Convert each keypoint to an array of x and y coordinates.
    const coordinates = keypoints.map(keypoint => [keypoint.x, keypoint.y]);

    // Stack the coordinates together into a single tensor.
    let tensor = tf.stack(coordinates);

    // Reshape the tensor to have rank 1.
    tensor = tf.reshape(tensor, [-1]);

    if (tensor.shape[0] !== INPUT_SIZE) {
        throw new Error(`Tensor length should be ${INPUT_SIZE}`);
    }

    tensor.print();

    return tensor;
}

function keypointsToArray(keypoints: Array<Keypoint>): Array<number> {
    return keypoints.reduce((acc: number[], keypoint: Keypoint) => {
        acc.push(keypoint.x);
        acc.push(keypoint.y);

        return acc;
    }, []);
}

const INPUT_SIZE = 34;

class PoseClassifier {

    posesData: PosesData = {};
    model: tf.Sequential;

    addPoseData(label: string, normalizedKeypoints: Keypoint[]) {
        if (!label) {
            throw new Error('Label is required');
        }

        if (!this.posesData[label]) {
            this.posesData[label] = [];
        }

        const posePoints: number[] = [];

        const torsoCenter = getTorsoCenter(normalizedKeypoints);

        // scale every keypoint to bounding box
        const scaledKeyPoints = scaleToBoundingBox(normalizedKeypoints);

        this.posesData[label].push(scaledKeyPoints);
    }

    async train() {
        const labelsSize = Object.keys(this.posesData).length;

        // todo refactor model provider to be dependency of a class
        this.model = createModel(INPUT_SIZE, labelsSize);
        this.model.summary();

        // iterate poses data and create tensors
        let trainingDataTansor: tf.Tensor[] = [];
        let featuresData: number[][] = [];
        let trainingLabels: number[] = [];

        Object.keys(this.posesData).forEach((label, index) => {
            this.posesData[label].forEach(poseData => {
                trainingDataTansor.push(keypointToTensor(poseData));
                featuresData.push(keypointsToArray(poseData));
                trainingLabels.push(index);
            });
        });

        const validationSplit = 0.15;
        await this.model.fit(
            tf.tensor2d(featuresData, [featuresData.length, INPUT_SIZE]),
            tf.oneHot(trainingLabels, labelsSize),
            {
                /*
                batchSize, */
                epochs: 10,
                validationSplit
            });
    }

    async predict(keypoints: Keypoint[]): Promise<string|undefined> {
        if (!this.model) {
            console.warn('Model is not trained yet');
            return;
        }

        const scaledKeyPoints = scaleToBoundingBox(keypoints);
        const newFeatureTensor = tf.tensor2d(keypointsToArray(scaledKeyPoints), [1, INPUT_SIZE]);
        const predictions = this.model.predict(newFeatureTensor) as tf.Tensor;

        console.log(
            'Predictions',
            Array.from(predictions.dataSync()),
        );
        
        const predictedLabelIndex = predictions.argMax(-1).dataSync()[0];

        return Object.keys(this.posesData)[predictedLabelIndex];
    }

    saveToStorage() {
        localStorage.setItem('posesData', JSON.stringify(this.posesData));
    }

    getFromStorage() {
        const posesData = localStorage.getItem('posesData');

        if (posesData && posesData.length > 0) {
            this.posesData = JSON.parse(posesData);
            console.log('Poses data loaded from local storage', this.posesData);
        } else {
            console.warn('No poses data found in local storage');
        }
    }
}

export { PoseClassifier };
import * as tf from '@tensorflow/tfjs-core';

import { Keypoint } from '@tensorflow-models/pose-detection/types'
import { getTorsoCenter, scaleToBoundingBox } from './pose-utils';

class PoseClassifier {

    posesData = {};

    addPoseData(label, normalizedKeypoints: Keypoint[]) {
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
        
        // collect posePoints and push them to posesData
        for (const keypoint of scaledKeyPoints) {
            posePoints.push(keypoint.x, keypoint.y);
        }

        this.posesData[label].push(posePoints);
    }

    saveToStorage() {
        localStorage.setItem('posesData', JSON.stringify(this.posesData));
    }

    getFromStorage() {
        const posesData = localStorage.getItem('posesData');

        if (posesData) {
            this.posesData = JSON.parse(posesData);
        }
    }
}

export { PoseClassifier };
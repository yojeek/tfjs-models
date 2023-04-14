import * as tf from '@tensorflow/tfjs-core';

import { Keypoint } from '@tensorflow-models/pose-detection/types'

type PosesData = {
    [label: string]: number[][]
}

enum BodyPart {
    NOSE = 'nose',
    LEFT_EYE = 'left_eye',
    RIGHT_EYE = 'right_eye',
    LEFT_EAR = 'left_ear',
    RIGHT_EAR = 'right_ear',
    LEFT_SHOULDER = 'left_shoulder',
    RIGHT_SHOULDER = 'right_shoulder',
    LEFT_ELBOW = 'left_elbow',
    RIGHT_ELBOW = 'right_elbow',
    LEFT_WRIST = 'left_wrist',
    RIGHT_WRIST = 'right_wrist',
    LEFT_HIP = 'left_hip',
    RIGHT_HIP = 'right_hip',
    LEFT_KNEE = 'left_knee',
    RIGHT_KNEE = 'right_knee',
    LEFT_ANKLE = 'left_ankle',
    RIGHT_ANKLE = 'right_ankle',
}

function getKeypointByBodyPart(keypoints: Keypoint[], bodyPart: BodyPart): Keypoint {
    const keypoint = keypoints.find(keypoint => keypoint.name === bodyPart);

    if (!keypoint) {
        throw new Error(`Keypoint ${bodyPart} not found`);
    }

    return keypoint;
}

function getTorsoCenter(keypoints: Keypoint[]) {
    const leftShoulder = getKeypointByBodyPart(keypoints, BodyPart.LEFT_SHOULDER);
    const rightShoulder = getKeypointByBodyPart(keypoints, BodyPart.RIGHT_SHOULDER);
    const leftHip = getKeypointByBodyPart(keypoints, BodyPart.LEFT_HIP);
    const rightHip = getKeypointByBodyPart(keypoints, BodyPart.RIGHT_HIP);

    const x = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4;
    const y = (leftShoulder.y + rightShoulder.y + leftHip.y + rightHip.y) / 4;

    return { x, y };
}

function getBoundingBox(keypoints: Keypoint[]) {
    const xMin = Math.min(...keypoints.map(keypoint => keypoint.x));
    const xMax = Math.max(...keypoints.map(keypoint => keypoint.x));
    const yMin = Math.min(...keypoints.map(keypoint => keypoint.y));
    const yMax = Math.max(...keypoints.map(keypoint => keypoint.y));

    return { xMin, xMax, yMin, yMax };
}

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
        const boundingBox = getBoundingBox(normalizedKeypoints);

        // scale every keypoint to bounding box
        for (let i = 0; i < normalizedKeypoints.length; i++) {
            const keypoint = normalizedKeypoints[i];
            const x = (keypoint.x - boundingBox.xMin) / (boundingBox.xMax - boundingBox.xMin);
            const y = (keypoint.y - boundingBox.yMin) / (boundingBox.yMax - boundingBox.yMin);
            normalizedKeypoints[i] = { ...keypoint, x, y };
        }

        // collect posePoints and push them to posesData
        for (let i = 0; i < normalizedKeypoints.length; i++) {
            const keypoint = normalizedKeypoints[i];
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
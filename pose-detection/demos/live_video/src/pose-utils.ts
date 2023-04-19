import { Keypoint } from "@tensorflow-models/pose-detection";

export enum BodyPart {
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

export const getBoundingBox = (keypoints: Keypoint[]) => {
    const xMin = Math.min(...keypoints.map(keypoint => keypoint.x));
    const xMax = Math.max(...keypoints.map(keypoint => keypoint.x));
    const yMin = Math.min(...keypoints.map(keypoint => keypoint.y));
    const yMax = Math.max(...keypoints.map(keypoint => keypoint.y));

    return { xMin, xMax, yMin, yMax };
}

export const scaleToBoundingBox = (keypoints: Keypoint[]) => {
    const boundingBox = getBoundingBox(keypoints);
    const { xMin, xMax, yMin, yMax } = boundingBox;

    return keypoints.map(keypoint => {
        const x = (keypoint.x - xMin) / (xMax - xMin);
        const y = (keypoint.y - yMin) / (yMax - yMin);
        return { ...keypoint, x, y };
    });
} 

export function getKeypointByBodyPart(keypoints: Keypoint[], bodyPart: BodyPart): Keypoint {
    const keypoint = keypoints.find(keypoint => keypoint.name === bodyPart);

    if (!keypoint) {
        throw new Error(`Keypoint ${bodyPart} not found`);
    }

    return keypoint;
}

export function getTorsoCenter(keypoints: Keypoint[]) {
    const leftShoulder = getKeypointByBodyPart(keypoints, BodyPart.LEFT_SHOULDER);
    const rightShoulder = getKeypointByBodyPart(keypoints, BodyPart.RIGHT_SHOULDER);
    const leftHip = getKeypointByBodyPart(keypoints, BodyPart.LEFT_HIP);
    const rightHip = getKeypointByBodyPart(keypoints, BodyPart.RIGHT_HIP);

    const x = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4;
    const y = (leftShoulder.y + rightShoulder.y + leftHip.y + rightHip.y) / 4;

    return { x, y };
}
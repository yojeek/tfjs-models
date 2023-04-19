import { Keypoint, calculators } from '@tensorflow-models/pose-detection';
import { ImageSize } from '@tensorflow-models/pose-detection/shared/calculators/interfaces/common_interfaces';
import OSC from 'osc-js';
import { scaleToBoundingBox } from './pose-utils';
import { log } from 'console';

type Pose = {
    score: number;
    keypoints: Keypoint[];
}

class OSCTransport {
    private osc: OSC;

    constructor() {
        this.osc = new OSC();
        this.osc.open();
    }

    public transmitPoses(poses: Pose[], frameSize: ImageSize, { scoreThreshold, scalePose = false, outputDebug}) {
        for (let i = 0; i < poses.length; i++) {
            const pose = poses[i];
            let keypoints = calculators.keypointsToNormalizedKeypoints(pose.keypoints, frameSize);

            if (scalePose) {
                keypoints = scaleToBoundingBox(keypoints);
            }

            if (outputDebug) {
                outputDebug.clearCtx();
                outputDebug.drawKeypoints(keypoints, true);
            }
            
            for (let j = 0; j < keypoints.length; j++) {
                const keypoint = keypoints[j];

                if (this.osc.status() === OSC.STATUS.IS_OPEN) {
                    if (keypoint.score && keypoint.score > scoreThreshold) {
                        const message = new OSC.Message(`/pose`, i, j, keypoint.x, keypoint.y, keypoint.name || 'unknown');
                        
                        this.osc.send(message);
                    }
                } else if (this.osc.status() === OSC.STATUS.IS_CLOSED) {
                    console.log('OSC connection closed, trying to reconnect...');
                    this.osc.open();
                }
            } 
        }
    }
}

export default OSCTransport;
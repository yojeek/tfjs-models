import { Keypoint, calculators } from '@tensorflow-models/pose-detection';
import { ImageSize } from '@tensorflow-models/pose-detection/shared/calculators/interfaces/common_interfaces';
import OSC from 'osc-js';

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

    public transmitPoses(poses: Pose[], frameSize: ImageSize, scoreThreshold: number) {
        for (let i = 0; i < poses.length; i++) {
            const pose = poses[i];
            const keypoints = calculators.keypointsToNormalizedKeypoints(pose.keypoints, frameSize);
            
            for (let j = 0; j < keypoints.length; j++) {
                const keypoint = keypoints[j];

                if (this.osc.status() === OSC.STATUS.IS_OPEN) {
                    if (keypoint.score && keypoint.score > scoreThreshold) {
                        this.osc.send(new OSC.Message(`/pose`, i, j, keypoint.x, keypoint.y, keypoint.name || 'unknown'));
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
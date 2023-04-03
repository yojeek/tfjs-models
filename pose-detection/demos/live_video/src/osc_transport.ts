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

    public transmitPoses(poses: Pose[], frameSize: ImageSize) {
        for (let i = 0; i < poses.length; i++) {
            const pose = poses[i];
            const keypoints = calculators.keypointsToNormalizedKeypoints(pose.keypoints, frameSize);
            
            for (let j = 0; j < keypoints.length; j++) {
                const keypoint = keypoints[j];

                this.osc.send(new OSC.Message(`/pose/`, j, keypoint.x, keypoint.y, keypoint.name || 'unknown'));
            }
        }
    }
}

export default OSCTransport;
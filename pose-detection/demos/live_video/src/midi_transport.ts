import { Keypoint, calculators } from '@tensorflow-models/pose-detection';
import { ImageSize } from '@tensorflow-models/pose-detection/shared/calculators/interfaces/common_interfaces';

type Pose = {
    score: number;
    keypoints: Keypoint[];
}

class MIDITransport {
    midiAccess: MIDIAccess;
    midiOutput: MIDIOutput;

    constructor() {
        navigator.requestMIDIAccess()
            .then(
                midiAccess => {
                    this.midiAccess = midiAccess
                    const outputs = midiAccess.outputs.values();

                    if (outputs) {
                        this.midiOutput = outputs.next().value;
                        console.log(this.midiOutput)
                    } else {
                        console.log('No MIDI output devices detected.')
                    }
                },
                onMIDIFailure
            );

        function onMIDIFailure() {
            console.log('Could not access your MIDI devices.');
        }
    }

    public transmitPoses(poses: Pose[], frameSize: ImageSize) {
        for (let i = 0; i < poses.length; i++) {
            const pose = poses[i];
            const keypoints = calculators.keypointsToNormalizedKeypoints(pose.keypoints, frameSize);

            const NOTE_ON = 0x90;
            const NOTE_OFF = 0x80;

            for (let j = 0; j < keypoints.length; j++) {
                const keypoint = keypoints[j];

                const floatToMidi = (float: number) => {
                    return Math.floor(float * 127);
                }
                
                // pick midi channel based on keypoint index
                const channelBase = j % 16 * 2;
                // send note on
                this.midiOutput.send([NOTE_ON + channelBase, floatToMidi(keypoint.x), 0x7f]);
                this.midiOutput.send([NOTE_ON + channelBase + 1, floatToMidi(keypoint.y), 0x7f]);
            }
        }
    }
}

export default MIDITransport;
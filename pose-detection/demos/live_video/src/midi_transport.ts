import { Keypoint, calculators } from '@tensorflow-models/pose-detection';
import { ImageSize } from '@tensorflow-models/pose-detection/shared/calculators/interfaces/common_interfaces';
import { log } from 'console';
import OSC from 'osc-js';

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
                
                this.midiOutput.send([NOTE_ON, floatToMidi(keypoint.x),  0x7f])
                // send midi message
            }
        }
    }
}

export default MIDITransport;
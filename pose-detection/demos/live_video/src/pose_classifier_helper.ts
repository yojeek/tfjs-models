import { PoseClassifier } from "./pose_classifier";

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

class PoseClassifierHelper {

    classifier: PoseClassifier
    currentPoseName: string
    STATE: any // global state

    constructor(STATE) {
        this.classifier = new PoseClassifier();
        this.STATE = STATE;

        this.currentPoseName = 'pose1';
    }

    addGuiElements(gui: any) {
        const classifierFolder = gui.addFolder('Classifier');

        classifierFolder.add(this, 'currentPoseName');
        classifierFolder.add(this, 'collect');

        classifierFolder.open();
    }

    async collect() {
        const overlayText = document.getElementById('overlay-text');

        // countdown 10 seconds
        async function countdown(seconds) {
            for (let i = seconds; i >= 0; i--) {
                overlayText.innerHTML = `Prepare to collect data in ${i} seconds`;
                await delay(1000);
            }
        }

        await countdown(0);

        const SAMPLE_COUNT = 1;

        // collect 10 samples
        for (let i = 0; i < SAMPLE_COUNT; i++) {
            overlayText.innerHTML = `Collecting ${i + 1}/10 sample`;
            this.classifier.addPoseData(this.currentPoseName, this.STATE.lastKeyPoints);
            await delay(500);
        }

        overlayText.innerHTML = '';
        console.log(this.classifier.posesData);
    }
}

export { PoseClassifierHelper };
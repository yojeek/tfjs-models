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
        classifierFolder.add(this, 'saveToStorage');
        classifierFolder.add(this, 'loadFromStorage');
        classifierFolder.add(this, 'train');

        classifierFolder.open();
    }

    saveToStorage() {
        this.classifier.saveToStorage();
    }

    loadFromStorage() {
        this.classifier.getFromStorage();
    }

    async train() {
        console.log('Loading poses data from storage.');
        this.classifier.getFromStorage();
        console.log('Training...');
        await this.classifier.train();
    }

    async predict() {
        const overlayText = document.getElementById('overlay-text');


        const keypoints = this.STATE.lastKeyPoints;
        if (keypoints && this.classifier.model) {
            const label = await this.classifier.predict(keypoints);
            console.log(`Predicted label: ${label}`);

            if (overlayText) {
                overlayText.innerHTML = 'Predicted label ' + label || '';
            }
        }
    }

    async collect() {
        const overlayText = document.getElementById('overlay-text');

        if (!overlayText) {
            console.error('overlay-text element not found');
            return;            
        }

        async function countdown(seconds) {
            for (let i = seconds; i >= 0; i--) {
                (overlayText as HTMLElement).innerHTML = `Prepare to collect data in ${i} seconds`;
                await delay(1000);
            }
        }

        await countdown(10);

        const SAMPLE_COUNT = 50;

        for (let i = 0; i < SAMPLE_COUNT; i++) {
            overlayText.innerHTML = `Collecting ${i + 1}/${SAMPLE_COUNT} sample`;
            this.classifier.addPoseData(this.currentPoseName, this.STATE.lastKeyPoints);
            await delay(50);
        }

        overlayText.innerHTML = '';
        console.log(this.classifier.posesData);
    }
}

export { PoseClassifierHelper };
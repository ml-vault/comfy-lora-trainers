import { app } from "../../scripts/app.js";

const getAction = (node, v) => {
    const currentInputs = node.inputs.length
    if (v > currentInputs) {
        return "add"
    }
    if (v < currentInputs) {
        return "remove"
    }
}

class AppManager {
    datasetCountWidgets = []
    constructor(){
        const self = this
        app.registerExtension({
            name: "a.unique.name.for.a.useless.extension",
            async nodeCreated(node) {
                if(node.title === "Dataset Loader Node") {
                    const [countWidget] = node.widgets
                    countWidget.callback = (v) => {
                        const currentInputs = node.inputs.length
                        const diff = Math.abs(v - currentInputs)
                        const action = getAction(node, v)
                        for (let i = 0; i < diff; i++) {
                            if(action === "add") {
                                node.addInput(`dataset-config-${i + 1 + currentInputs}`, "KHXL_CF_DATASET")
                            }
                            if(action === "remove") {
                                node.removeInput(-1)
                            }
                        }
                    }
                } else if (node.title === "DreamBooth Dataset Config Node") {
                    const countWidget = node.widgets.find(widget => widget.name === "dream_booth_subset_count")
                    console.log(countWidget)
                    countWidget.callback = (v) => {
                        const currentInputs = node.inputs.length
                        const diff = Math.abs(v - currentInputs)
                        const action = getAction(node, v)
                        for (let i = 0; i < diff; i++) {
                            if(action === "add") {
                                node.addInput(`dream-booth-subset-config-${i + 1 + currentInputs}`, "KHXL_CF_DREAMBOOTH_SUBSET_CONFIG")
                            }
                            if(action === "remove") {
                                node.removeInput(-1)
                            }
                        }
                    }
                }
            },
        })
    }
}

new AppManager()

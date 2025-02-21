import { app } from "../../scripts/app.js";

class DatasetInputWidget {
    constructor(node, index) {
        this.node = node
        this.index = index
        this.id = `dataset_${index + 1}`
    }
    build() {
        this.input = this.node.addInput(this.id, "KHXL_CF_DATASET")
    }
    remove() {
        const index = this.node.findInputSlot(this.id)
        this.node.removeInput(index)
    }
    callback(value) {
        this.input.value = value
    }
}

class DatasetCountWidget {
    widgets = []

    constructor(node, oldval = 1) {
        this.node = node
        this.value = oldval
        this.widget = this.node.addWidget("number",
            "dataset_count",
            this.value, (v) =>this.callback(v, this.value), {
                step: 10,
                default: 1,
                round: 1,
                precision: 0,
                min: 1,
                max: 1000000
            }
        )
        this.callback(1)
    }
    callback(newval) {
        const existingWidgets = this.widgets.length
        const isIncreased = newval > existingWidgets
        const toAdd = isIncreased ? newval - existingWidgets : 0
        if(toAdd) {
            const newItems = Array.from({length: toAdd}, (_, i) => new DatasetInputWidget(this.node, i + existingWidgets))
            newItems.forEach(item => item.build())
            this.widgets.push(...newItems)
        }

        const isDecreased = newval < existingWidgets
        const toRemove = isDecreased ? existingWidgets - newval : 0
        if(toRemove) {
            const removedItems = this.widgets.slice(-toRemove)
            removedItems.forEach(item => item.remove())
            this.widgets = this.widgets.slice(0, -toRemove)
        }
        console.log(this.widgets)
        this.value = newval
    }
}
app.registerExtension({
	name: "a.unique.name.for.a.useless.extension",
	async setup() {
		alert("Setup complete!")
	},
    async nodeCreated(node) {
        console.log(node)
        if(node.title === "Dataset Loader Node") {
            new DatasetCountWidget(node)
        }
    }
})

import { ModelTaskType } from "./types";


export function getModelPath(taskType: ModelTaskType){
    switch(taskType){
        case ModelTaskType.DETECT:
            return `./models/detect`
        case ModelTaskType.ORIENTED:
            return `./models/obb`
        case ModelTaskType.SEGMENT:
            return `./models/seg`
    }
}

export function getImagePath(taskType: ModelTaskType){
    switch(taskType){
        case ModelTaskType.DETECT:
            return "./images/detect.jpg"
        case ModelTaskType.ORIENTED:
            return "./images/obb.jpg"
        case ModelTaskType.SEGMENT:
            return "./images/seg.jpg"
    }
}
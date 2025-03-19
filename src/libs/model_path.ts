import { ModelTaskType } from "./types";

export function getModelPath(taskType: ModelTaskType) {
  switch (taskType) {
    case ModelTaskType.DETECT:
      return `./models/detect`;
    case ModelTaskType.ORIENTED:
      return `./models/obb`;
    case ModelTaskType.SEGMENT:
      return `./models/seg`;
    case ModelTaskType.V12_DETECT:
      return `./models/v12_detect`;
  }
}

export function getImagePath(taskType: ModelTaskType) {
  switch (taskType) {
    case ModelTaskType.DETECT:
      return "./images/detect.jpg";
    case ModelTaskType.ORIENTED:
      return "./images/obb.jpg";
    case ModelTaskType.SEGMENT:
      return "./images/seg.jpg";
    case ModelTaskType.V12_DETECT:
      return "./images/detect.jpg";
  }
}

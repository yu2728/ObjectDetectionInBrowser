import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";
import { convertImageElement, tensorFromPixel } from "./conver_image_element";
import { DetectBbox } from "./types";

/**
 * 物体検出のBBOXに変換する(YOLO v12)
 * @param model
 * @param image
 * @param imgsz
 * @param minScore
 * @returns
 */
export const detectV12 = async (
  model: tf.GraphModel<string | tf.io.IOHandler>,
  image: HTMLImageElement,
  imgsz: [number, number],
  minScore: number
): Promise<DetectBbox[]> => {
  const convertedCanvas = convertImageElement(image, imgsz);
  const imageTensor = tensorFromPixel(convertedCanvas, imgsz);
  const result = await model.predictAsync(imageTensor) as tf.Tensor2D[];
  const mask = result[0].squeeze().slice([0, 4], [-1, 1]).greater(minScore).squeeze();
  const filteredBbox = await tf.booleanMaskAsync(result[0].squeeze(), mask);
  const bboxes = Object.entries(filteredBbox.arraySync()).map(e => {
    const x1 = e[1][0];
    const y1 = e[1][1];
    const x2 = e[1][2];
    const y2 = e[1][3];
    return {x: x1, y: y1, w: x2 - x1, h: y2 - y1, score: e[1][4], label: e[1][5]} as DetectBbox;
  })
  imageTensor.dispose();
  mask.dispose();
  filteredBbox.dispose();
  result.forEach(e => e.dispose());
  return bboxes;
};

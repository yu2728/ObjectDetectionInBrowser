import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";
import { convertImageElement, tensorFromPixel } from "./conver_image_element";
import { OrientedBbox } from "./types";

/**
 * OrientedBboxに変換する
 * @param model
 * @param image
 * @param imgsz
 * @param minScore
 * @returns 
 */
export const obb = async (
  model: tf.GraphModel<string | tf.io.IOHandler>,
  image: HTMLImageElement,
  imgsz: [number, number],
  minScore: number,
  ): Promise<OrientedBbox[]> => {
      const convertedCanvas = convertImageElement(image, imgsz);
      const imageTensor = tensorFromPixel(convertedCanvas, imgsz);
      const result = await model.predictAsync(imageTensor) as tf.Tensor2D[];
      const mask = result[0].squeeze().slice([0, 4], [-1, 1]).greater(minScore).squeeze();
      const filteredBbox = await tf.booleanMaskAsync(result[0].squeeze(), mask);
      const bboxes = Object.entries(filteredBbox.arraySync()).map(e => {
        const cx = e[1][0];
        const cy = e[1][1];
        const w = e[1][2];
        const h = e[1][3];
        const r = e[1][6];
        return {x: cx, y: cy, w: w, h: h, score: e[1][4], label: e[1][5], r: r} as OrientedBbox;
      })
      imageTensor.dispose();
      mask.dispose();
      filteredBbox.dispose();
      result.forEach(e => e.dispose());

    return bboxes;
  };
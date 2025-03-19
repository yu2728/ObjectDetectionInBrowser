import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";
import { convertImageElement, tensorFromPixel } from "./conver_image_element";
import { SegBbox } from "./types";

/**
 * SegmentationBBOXに変換する
 * @param model
 * @param image
 * @param imgsz
 * @param minScore
 * @returns
 */
export const seg = async (model: tf.GraphModel<string | tf.io.IOHandler>, image: HTMLImageElement, imgsz: [number, number], minScore: number): Promise<SegBbox[]> => {
  const convertedCanvas = convertImageElement(image, imgsz);
  const imageTensor = tensorFromPixel(convertedCanvas, imgsz);
  const result = (await model.predictAsync(imageTensor)) as tf.Tensor2D[];
  const boxIndexes = result[0].squeeze().slice([0, 4], [-1, 1]).greater(minScore).squeeze();
  const filteredBbox = await tf.booleanMaskAsync(result[0].squeeze(), boxIndexes);

  // get Mask
  // 予測したボックスのマスクを取り出す
  const vectors = filteredBbox.slice([0, 6], [-1, -1]);
  // 画像を一つの配列に変換
  const maskWeight = result[2].squeeze().reshape([160 * 160, 32]);
  // 変換
  const transponsedVectors = vectors.transpose([1, 0]);
  // マスクの重みとベクトルの内積を取る
  const dotProduct = tf.matMul(maskWeight, transponsedVectors);
  // シグモイド関数で0から1の範囲に変換
  const probabiltyMap = dotProduct.sigmoid();
  // minScore以上の確率を1、それ以外を0とする
  const binaryMask = probabiltyMap.greater(minScore);
  const masks = binaryMask.transpose([1, 0]).reshape([filteredBbox.shape[0], 160, 160]).arraySync() as [];
  const bboxes = Object.entries(filteredBbox.arraySync()).map((e, index) => {
    const x1 = e[1][0];
    const y1 = e[1][1];
    const x2 = e[1][2];
    const y2 = e[1][3];
    return { x: x1, y: y1, w: x2 - x1, h: y2 - y1, score: e[1][4], label: e[1][5], mask: masks[index] } as SegBbox;
  });
  imageTensor.dispose();
  filteredBbox.dispose();
  result.forEach((e) => e.dispose());
  return bboxes;
};

import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";

/**
 * YOLOのモデルをロードし、初期化を行う
 * @param {string} path モデルが保存されているフォルダパス.最後にスラッシュはつけない
 * @param {[number, number]} imgsz metadataで取得した画像サイズ
 * @returns {Promise<tf.GraphModel<string | tf.io.IOHandler>>} モデルをロードするPromiseを返す
 */
export async function loadYOLOModel(path: string, imgsz: [number, number]): Promise<tf.GraphModel<string | tf.io.IOHandler>> {
  const model = await tf.loadGraphModel(`${path}/model.json`);
  // warm up
  const zeroTensor = tf.zeros([1, imgsz[0], imgsz[1], 3], "float32");
  await model.executeAsync(zeroTensor);
  zeroTensor.dispose();
  return model;
}

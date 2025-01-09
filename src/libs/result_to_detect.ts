import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";
import { DetectBbox } from "./types";

/**
 * 物体検出のBBOXに変換する
 * @param result
 * @param labelCount
 * @param maxOutputSize
 * @param iouThreshold
 * @param minScore
 * @param targetId
 * @returns
 */
export const resultToDetectBbox = (result: tf.Tensor<tf.Rank>, labelCount: number, maxOutputSize: number, iouThreshold: number, minScore: number, targetId?: number | undefined): DetectBbox[] => {
  const bbox = tf.tidy(() => {
    const temp = result.squeeze();
    // x, y, w, hを取り出し、[y1, x1, y2, x2]形式に変換
    const x = temp.slice([0, 0], [1, -1]);
    const y = temp.slice([1, 0], [1, -1]);
    const w = temp.slice([2, 0], [1, -1]);
    const h = temp.slice([3, 0], [1, -1]);
    const x1 = tf.sub(x, tf.div(w, 2));
    const y1 = tf.sub(y, tf.div(h, 2));
    const x2 = tf.add(x1, w);
    const y2 = tf.add(y1, h);
    const boxes = tf.stack([y1, x1, y2, x2], 2).squeeze();

    // スコアが一番いいラベルのスコア一覧を取得。 
    const maxScores = targetId === undefined ? temp.slice([4, 0], [labelCount, -1]).max(0) : temp.slice([targetId + 4, 0], [1, -1]).max(0);
    // スコアが一番いいラベルのインデックス一覧を取得。 
    const labelIndexes = targetId === undefined ? temp.slice([4, 0], [labelCount, -1]).argMax(0) : tf.fill(maxScores.shape, targetId);
    // NMSで不要なデータを削除。インデックス一覧を取得
    const boxIndexes = tf.image.nonMaxSuppression(boxes.as2D(boxes.shape[0], boxes.shape[1]!), maxScores.as1D(), maxOutputSize, iouThreshold, minScore);

    // NMSで取得したインデックスを基にデータを絞り込み
    const resultBboxes = boxes.gather(boxIndexes, 0).arraySync() as [];
    const resultScores = maxScores.gather(boxIndexes, 0).arraySync() as [];
    const resultLabels = labelIndexes.gather(boxIndexes, 0).arraySync() as [];
    // DetectBbox[]形式で結果を返す
    return resultBboxes.map((bbox, index) => {
      return {
        x: bbox[1],
        y: bbox[0],
        w: bbox[3] - bbox[1],
        h: bbox[2] - bbox[0],
        score: resultScores[index],
        label: resultLabels[index],
      };
    });
  });
  return bbox as DetectBbox[];
};

import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";
import { resultToDetectBbox } from "./result_to_detect";
import { resultToOrientedBbox } from "./result_to_obb";
import { resultToSegBbox } from "./result_to_seg";
import { DetectBbox, ModelTaskType, OrientedBbox, SegBbox } from "./types";

/**
 * 推論を行い、対応したBBOXを返す
 * @param {ModelTaskType} taskType モデルのタスクタイプ
 * @param {tf.GraphModel<string | tf.io.IOHandler>} model YOLOのモデル
 * @param {tf.Tensor} imageTensor 推論する画像のTensor
 * @param {number} labelCount ラベルの数
 * @param {number} minScore 検出する最低スコア
 * @param {number | undefined} targetId 絞りこむときはラベルのIDを指定
 * @returns {DetectBbox[] | SegBbox[] | OrientedBbox[]} BBOXの配列
 */
export const predict = (
  taskType: ModelTaskType,
  model: tf.GraphModel<string | tf.io.IOHandler>,
  imageTensor: tf.Tensor,
  labelCount: number,
  minScore: number,
  targetId: number | undefined
): DetectBbox[] | SegBbox[] | OrientedBbox[] => {
  const maxOutputSize = 200;
  const iouThreshold = 0.5;
  const bbox: DetectBbox[] | SegBbox[] | OrientedBbox[] = tf.tidy(() => {
    switch (taskType) {
      case ModelTaskType.DETECT: {
        const result = model!.predict(imageTensor) as tf.Tensor<tf.Rank>;
        return resultToDetectBbox(result, labelCount, maxOutputSize, iouThreshold, minScore, targetId) as [];
      }
      case ModelTaskType.ORIENTED: {
        const result = model!.predict(imageTensor) as tf.Tensor<tf.Rank>;
        return resultToOrientedBbox(result, labelCount, maxOutputSize, iouThreshold, minScore, targetId) as [];
      }
      case ModelTaskType.SEGMENT: {
        const result = model!.predict(imageTensor) as tf.Tensor<tf.Rank>[];
        return resultToSegBbox(result, labelCount, maxOutputSize, iouThreshold, minScore, targetId) as [];
      }
    }
  });
  return bbox;
};

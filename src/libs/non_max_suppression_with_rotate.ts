
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";
import * as turf from "@turf/turf";
import { Feature, GeoJsonProperties, Polygon } from 'geojson';

/**
 * Bboxの候補
 */
interface Candidate {
  score: number
  boxIndex: number
  box: Feature<Polygon, GeoJsonProperties> | null
}

/**
 * Rorateに対応したNonMaxSuppression
 * (重複する検出結果の中から最も信頼度の高いものを選択し、他を除去する手法)
 * @param {tf.Tensor2D} boxes 検出した物体のBBox
 * @param {tf.Tensor1D} score 検出した物体のスコア
 * @param {number} maxOutputSize 
 * @param {number} iouThreshold IoUの閾値
 * @param {number} scoreThreshold スコアの閾値
 * @returns {tf.Tensor1D} 重複を削除されたボックスのインデックス
 */
export default function nonMaxSuppressionWithRotate(
    boxes: tf.Tensor2D,
    score: tf.Tensor1D,
    maxOutputSize: number,
    iouThreshold: number = 0.5,
    scoreThreshold: number = 0.3
  ): tf.Tensor1D {
    let candidates: Candidate[] = []
    // Scoreの閾値以下を切り捨て。結果はindexで返すためCandidate型に変換し、元のindexを保持する
    const scoreArray = score.arraySync()
    for (let i = 0; i < scoreArray.length; i++) {
      if (scoreArray[i] > scoreThreshold) {
        candidates.push({ score: scoreArray[i], boxIndex: i, box: null } as Candidate)
      }
    }
  
    // scoreがいい順番に並び変え
    candidates.sort((a, b) => (b.score - a.score))

    // 回転した座標に変換する
    const candidatesTensor = tf.tensor1d(candidates.map(e => e.boxIndex), "int32")
    const rotatedMatrix = rotationMatrix(boxes.gather(candidatesTensor, 0))

    // turfで処理するためのポリゴンに変更
    const polygons = matrix2Polygons(rotatedMatrix)
    polygons.forEach((polygon, index) => {
      candidates[index].box = polygon
    })
  
    // 選択されたボックスのインデックスを格納する配列を初期化
    const selectedindexes: Candidate[] = [];

    // ボックスの重複を削除する処理
    while (candidates.length > 0) {
      const currentCandidate = candidates[0]
      // 残っている候補で一番いいスコアのboxは残す
      selectedindexes.push(candidates[0])
      // maxを超えていたら終わり
      if (selectedindexes.length >= maxOutputSize) {
        break;
      }
      // 一番いいスコアのboxと残っているboxを比較し、IoUの値が閾値より小さいもののみ候補に残す
      candidates.filter(box => box.boxIndex !== currentCandidate.boxIndex)
      candidates = candidates.filter((candidate) => {
        if (candidate.boxIndex === currentCandidate.boxIndex) return false;
        const iou = calculateRotatedIOU(currentCandidate.box!, candidate.box!)
        return iou < iouThreshold
      })
    }
    // 重複を削除した結果のボックスインデックスを返す
    return tf.tensor1d(selectedindexes.map(e => e.boxIndex), "int32")
  }
  
  /**
   * tarf.jsのFeature<Polygon, GeoJsonProperties>を元に
   * 二つのBBOXのIoUを計算する
   * @param {Feature<Polygon, GeoJsonProperties>} polygon_a 
   * @param {Feature<Polygon, GeoJsonProperties>} polygon_b 
   * @returns {number} IoUの数値
   */
  function calculateRotatedIOU(polygon_a: Feature<Polygon, GeoJsonProperties>, polygon_b: Feature<Polygon, GeoJsonProperties>): number {
    const intersectPolygon = turf.intersect(turf.featureCollection([polygon_a, polygon_b]))
    if (!intersectPolygon) {
      return 0
    }
    const unionPolygon = turf.union(turf.featureCollection([polygon_a, polygon_b]))
  
    if (!unionPolygon) {
      return 0
    }
  
    const iou = turf.area(intersectPolygon) / turf.area(unionPolygon)
    return iou
  
  }
  
  /**
   * bboxとradを元に回転した座標を求める
   * @param {tf.Tensor<tf.Rank>} boxes 検出結果
   * @returns {tf.Tensor<tf.Rank>} 回転した座標
   */
  function rotationMatrix(boxes: tf.Tensor<tf.Rank>) {
    const results = tf.tidy(() => {
      const [x, y, w, h, rad] = tf.split(boxes, [1, 1, 1, 1, 1], 1)
      // cosAとsinAを計算 (角度のラジアン部分)
      const cos = tf.cos(rad).squeeze();
      const sin = tf.sin(rad).squeeze();
  
      // x,yを中心としてx1~4, y1~4を求める
      const x1 = w.div(-2).squeeze()
      const x2 = w.div(2).squeeze()
      const y1 = h.div(-2).squeeze()
      const y2 = h.div(2).squeeze()
      // 回転した点を求めるp1~時計回りに進める
      const p1x = x1.mul(cos).sub(y1.mul(sin)).add(x.squeeze())
      const p1y = x1.mul(sin).add(y1.mul(cos)).add(y.squeeze())
      const p2x = x2.mul(cos).sub(y1.mul(sin)).add(x.squeeze())
      const p2y = x2.mul(sin).add(y1.mul(cos)).add(y.squeeze())
      const p3x = x2.mul(cos).sub(y2.mul(sin)).add(x.squeeze())
      const p3y = x2.mul(sin).add(y2.mul(cos)).add(y.squeeze())
      const p4x = x1.mul(cos).sub(y2.mul(sin)).add(x.squeeze())
      const p4y = x1.mul(sin).add(y2.mul(cos)).add(y.squeeze())
  
      return tf.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y])
    })
    return results
  }
  
  /**
   * tf.Tensor型のBBox行列をturf.jsのFeature<Polygon, GeoJsonProperties>の配列に変換する
   * @param {tf.Tensor<tf.Rank>} matrix 回転した座標の行列
   * @returns {Feature<Polygon, GeoJsonProperties>[]} tarf.jsのFeature<Polygon, GeoJsonProperties>の配列
   */
  function matrix2Polygons(matrix: tf.Tensor<tf.Rank>): Feature<Polygon, GeoJsonProperties>[] {
  
    const transposedMatrix = tf.tidy(() => {
      return matrix.transpose([1, 0])
  
    })
    const matrixArray = transposedMatrix.arraySync() as []
    return matrixArray.map(e => {
      return turf.polygon([[
        [e[0], e[1]],
        [e[2], e[3]],
        [e[4], e[5]],
        [e[6], e[7]],
        [e[0], e[1]],
      ]])
    })
  }
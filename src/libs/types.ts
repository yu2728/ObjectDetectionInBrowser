/**
 * モデルの種類
 */
export enum ModelTaskType {
  DETECT = "detect",
  SEGMENT = "segment",
  ORIENTED = "oriented",
  POSE = "pose",
  V12_DETECT = "v12_detect",
}

/**
 * YOLOのメタデータのデータ型
 */
export interface YOLOMetadata {
  description: string;
  author: string;
  date: string;
  version: string;
  license: string;
  docs: string;
  stride: number;
  task: ModelTaskType;
  batch: number;
  imgsz: [number, number];
  names: { [key: number]: string };
  args: Args;
}

interface Args {
  batch: number;
  half: boolean;
  int8: boolean;
  nms: boolean;
}

/**
 * 物体検出のバウンディングボックス
 */
export interface DetectBbox {
  x: number;
  y: number;
  w: number;
  h: number;
  label: number;
  score: number;
}

/**
 * セグメンテーションのバウンディングボックス
 */
export interface SegBbox {
  x: number;
  y: number;
  w: number;
  h: number;
  label: number;
  score: number;
  mask: number[][];
}

/**
 * Oriented Bounding Box
 */
export interface OrientedBbox {
  x: number;
  y: number;
  w: number;
  h: number;
  label: number;
  score: number;
  r: number;
}

/**
 * POSE Bounding Box
 */
export interface PoseBbox {
  x: number;
  y: number;
  w: number;
  h: number;
  label: number;
  score: number;
  r: number;
  keypoints: PoseKeypoint[];
}

/**
 * https://docs.ultralytics.com/ja/tasks/pose/
 * 上記のURLから
 * 0 鼻
 * 1 左目
 * 2 右目
 * 3 左耳
 * 4 右耳
 * 5 左肩
 * 6 右肩
 * 7 左肘
 * 8 右肘
 * 9 左手首
 * 10 右手首
 * 11 左ヒップ
 * 12 右ヒップ
 * 13 左膝
 * 14 右膝
 * 15 左足首
 * 16 右足首 
 */
export enum Keypoint {
  Nose,
  LeftEye,
  RightEye,
  LeftEar,
  RightEar,
  LeftShoulder,
  RightShoulder,
  LeftElbow,
  RightElbow,
  LeftWrist,
  RightWrist,
  LeftHip,
  RightHip,
  LeftKnee,
  RightKnee,
  LeftAnkle,
  RightAnkle
}

export interface PoseKeypoint{
  x: number;
  y: number;
  score: number;
  keypoint: Keypoint
}
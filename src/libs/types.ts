/**
 * モデルの種類
 */
export enum ModelTaskType {
  DETECT = "detect",
  SEGMENT = "segment",
  ORIENTED = "oriented",
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

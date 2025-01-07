
/**
 * モデルの種類
 */
export enum ModelTaskType {
    DETECT = 'detect',
    SEGMENT = 'segment',
    ORIENTED = 'oriented'
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
}



/**
 * Bounding Boxのデータ型
 */
export interface BaseBbox {
    x: number;
    y: number;
    w: number;
    h: number;
    label: number;
    score: number;
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
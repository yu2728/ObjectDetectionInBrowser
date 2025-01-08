import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import { SegBbox } from "../types";

const colors: number[][] = [
  [204, 153, 255], // 紫系
  [255, 153, 204], // ピンク系
  [153, 255, 238], // エメラルドグリーン系
  [255, 221, 153], // オレンジ系
  [153, 170, 255], // 青紫系
  [255, 153, 153], // 赤系
  [153, 221, 255], // 水色系
  [255, 170, 153], // コーラル系
  [153, 255, 187], // ライトグリーン系
  [221, 255, 153], // 黄緑系
];

export function segmentView(imageCanvas: HTMLCanvasElement, maskCanvas: HTMLCanvasElement, image: HTMLImageElement, imgsz: [number, number], bboxes: SegBbox[]) {
  const processCtx = maskCanvas.getContext("2d", {
    willReadFrequently: true,
  }) as CanvasRenderingContext2D;

  maskCanvas.width = 160;
  maskCanvas.height = 160;

  processCtx.fillStyle = "black"; // 塗りつぶす色を黒に設定
  processCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

  const scale1 = 160 / imgsz[0]
  bboxes.forEach((bbox, index) => {
    bbox.mask.forEach((col, colIndex) => {
      col.forEach((pixel, rowIndex) => {
        if (pixel === 1) {
          // bbox内にあるマスクのみ設定する
          if(
            (bbox.x * scale1 <= rowIndex && rowIndex <= (bbox.x + bbox.w) * scale1) &&
            (bbox.y * scale1 <= colIndex && colIndex <= (bbox.y + bbox.h) * scale1)
          ){
            const imageData = processCtx.getImageData(rowIndex, colIndex, 1, 1);
            imageData.data[0] = index + 1; // Rの数値にindex + 1を入れる
            processCtx.putImageData(imageData, rowIndex, colIndex); // 変更したピクセルデータを反映
          }
        }
      });
    });
  });

  const imageData = processCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

  maskCanvas.width = image.naturalWidth;
  maskCanvas.height = image.naturalHeight;
  // 拡大して再描画
  processCtx.imageSmoothingEnabled = true;
  processCtx.putImageData(imageData, 0, 0);
  const scale = Math.min(imgsz[0] / image.naturalWidth, imgsz[1] / image.naturalHeight);
  const clip =
    image.naturalWidth > image.naturalHeight
      ? [imageData.width, imageData.height * ((image.naturalHeight * scale) / imgsz[1])]
      : [imageData.width * ((image.naturalWidth * scale) / imgsz[0]), imageData.height];
  processCtx.drawImage(maskCanvas, 0, 0, clip[0], clip[1], 0, 0, maskCanvas.width, maskCanvas.height);

  // マスクデータ
  const processData = processCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

  // canvasに表示
  const viewCtx = imageCanvas.getContext("2d", {
    willReadFrequently: true,
  }) as CanvasRenderingContext2D;

  imageCanvas.width = image.naturalWidth;
  imageCanvas.height = image.naturalHeight;

  viewCtx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);

  const viewData = viewCtx.getImageData(0, 0, imageCanvas.width, imageCanvas.height);

  const alphaBlend = 0.6;
  for (let i = 0; i < processData.data.length; i += 4) {
    const r = processData.data[i];
    if (r !== 0) {
      const color = colors[r % 10];
      viewData.data[i] = Math.floor(viewData.data[i] * (1 - alphaBlend) + color[0] * alphaBlend); // R
      viewData.data[i + 1] = Math.floor(viewData.data[i + 1] * (1 - alphaBlend) + color[1] * alphaBlend); // G
      viewData.data[i + 2] = Math.floor(viewData.data[i + 2] * (1 - alphaBlend) + color[2] * alphaBlend); // B
      viewData.data[i + 3] = 255; // 不透明のまま
    }
  }
  viewCtx.putImageData(viewData, 0, 0);

  const restoreScale = (1 / scale)
  bboxes.forEach((bbox, index) => {
    const color = colors[(index + 1) % 10]
    viewCtx.beginPath();
    viewCtx.rect(bbox.x * restoreScale, bbox.y * restoreScale, bbox.w * restoreScale, bbox.h * restoreScale);
    viewCtx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    viewCtx.lineWidth = 5 * ((imageCanvas.width ?? 640) / 640);
    viewCtx.stroke();
  });
}

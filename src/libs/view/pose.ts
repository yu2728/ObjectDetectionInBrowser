import { Keypoint, PoseBbox } from "../types";

export function poseView(canvas: HTMLCanvasElement, image: HTMLImageElement, restoreScale: number, bboxes: PoseBbox[]) {
  const context = canvas.getContext("2d") as CanvasRenderingContext2D;

  canvas.width = image.width;
  canvas.height = image.height;

  context.drawImage(image, 0, 0, image.width, image.height);

  context.font = "10px Arial";

  bboxes.forEach((bbox) => {
    context.beginPath();
    context.rect(bbox.x * restoreScale, bbox.y * restoreScale, bbox.w * restoreScale, bbox.h * restoreScale);
    context.strokeStyle = "salmon";
    context.lineWidth = 4 * ((canvas.width ?? 640) / 640);
    context.stroke();

    bbox.keypoints.map((keypoint) => {
      context.beginPath();
      context.arc(keypoint.x * restoreScale, keypoint.y * restoreScale, 3, 0, Math.PI * 2); // (0,0) が中心
      context.fillStyle = "red"; // 中心点の色
      context.fill();
      const keypointName = Keypoint[keypoint.keypoint];

      // テキストの座標を設定（少し右上にオフセット）
      const textX = keypoint.x * restoreScale + 5;
      const textY = keypoint.y * restoreScale - 5;
      context.font = "12px Arial bold";
      context.lineWidth = 3; // 縁取りの太さ
      context.strokeStyle = "white"; // 縁取りの色
      context.strokeText(keypointName, textX, textY);

      // テキスト本体を描画
      context.fillStyle = "black"; // 文字色
      context.fillText(keypointName, textX, textY);
    });
  });
}

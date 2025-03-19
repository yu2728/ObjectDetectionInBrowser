import { OrientedBbox } from "../types";

export function orientedView(canvas: HTMLCanvasElement, image: HTMLImageElement, restoreScale: number, bboxes: OrientedBbox[]){
    const context = canvas.getContext("2d") as CanvasRenderingContext2D;
    
    canvas.width = image.width
    canvas.height = image.height

    context.drawImage(image, 0, 0, image.width, image.height)

    bboxes.forEach(bbox => {
      context.save();
  
      // BBoxのスケール変換後の値
      const x = bbox.x * restoreScale;
      const y = bbox.y * restoreScale;
      const w = bbox.w * restoreScale;
      const h = bbox.h * restoreScale;
  
      // 中心座標
      const centerX = x;
      const centerY = y;  
      // 1. 中心を基準に座標を移動し回転
      context.translate(centerX, centerY);
      context.rotate(bbox.r);
  
      // 2. BBoxの枠を描画（回転後の座標）
      context.beginPath();
      context.rect(-w / 2, -h / 2, w, h); // (0,0) を BBoxの中心にして描画
      context.strokeStyle = "cyan";
      context.lineWidth = 2;
      context.stroke();
  
      // 3. BBoxの中心点に円を描画
      context.beginPath();
      context.arc(0, 0, 2, 0, Math.PI * 2); // (0,0) が中心
      context.fillStyle = "red"; // 中心点の色
      context.fill();
  
      context.restore();
  });
  

}

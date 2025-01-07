import { SegBbox } from "../types";


export function segmentView(canvas: HTMLCanvasElement, image: HTMLImageElement, restoreScale: number, bboxes: SegBbox[]){
    const context = canvas.getContext("2d") as CanvasRenderingContext2D;
    
    canvas.width = image.width
    canvas.height = image.height

    context.drawImage(image, 0, 0, image.width, image.height)

    context.font = "10px Arial";

    bboxes.forEach((bbox) => {
      context.beginPath();
      context.rect(
        bbox.x * restoreScale,
        bbox.y * restoreScale,
        bbox.w * restoreScale,
        bbox.h * restoreScale
      );
      context.strokeStyle = "salmon";
      context.lineWidth = 4 * ((canvas.width ?? 640) / 640);
      context.stroke();
    });

}
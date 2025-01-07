import { OrientedBbox } from "../types";

export function orientedView(canvas: HTMLCanvasElement, image: HTMLImageElement, restoreScale: number, bboxes: OrientedBbox[]){
    const context = canvas.getContext("2d") as CanvasRenderingContext2D;
    
    canvas.width = image.width
    canvas.height = image.height

    context.drawImage(image, 0, 0, image.width, image.height)

    bboxes.forEach(bbox => {
        context.save()
        context.translate(
          bbox.x * restoreScale + ((bbox.w * restoreScale) / 2),
          bbox.y * restoreScale + ((bbox.h * restoreScale) / 2)
        )
        context.rotate(bbox.r)
        context.beginPath();
        context.rect(
          -bbox.w * restoreScale / 2,
          -bbox.h * restoreScale / 2,
          bbox.w * restoreScale,
          bbox.h * restoreScale)
        context.strokeStyle = "cyan"
        context.lineWidth = 2
        context.stroke()
        context.font = '20px Roboto medium';
        context.fillStyle = 'cyan';
  
        context.restore()
      })

}

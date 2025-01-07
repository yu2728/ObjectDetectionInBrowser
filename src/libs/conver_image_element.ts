import * as tf from "@tensorflow/tfjs";

/**
 * ImageElementをimgszのCanvasに変換します。
 * 縦横比を維持したままリサイズし、足りない部分は黒で埋めます。
 * @param {HTMLImageElement | HTMLCanvasElement | HTMLVideoElement} image 変換したい画像の要素
 * @param imgsz YOLOのmetadataで取得した画像サイズ
 * @returns {HTMLCanvasElement} 変換後のCanvas要素
 */
export const convertImageElement = (
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  imgsz: [number, number]
) => {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas context is not available");
  }
  canvas.width = imgsz[0];
  canvas.height = imgsz[1];

  const originalWidth =
    image instanceof HTMLVideoElement ? image.videoWidth : image.width;
  const originalHeight =
    image instanceof HTMLVideoElement ? image.videoHeight : image.height;

  const scale = Math.min(imgsz[0] / originalWidth, imgsz[1] / originalHeight);

  const newWidth = originalWidth * scale;
  const newHeight = originalHeight * scale;

  context.fillStyle = "black";
  context.fillRect(0, 0, imgsz[0], imgsz[1]);

  context.drawImage(image, 0, 0, newWidth, newHeight);

  return canvas;
};

/**
 * ImageElementをTensorに変換します。
 * @param {HTMLImageElement | HTMLCanvasElement | HTMLVideoElement} image 変換したい画像の要素
 * @param imgsz YOLOのmetadataで取得した画像サイズ
 * @returns {tf.Tensor<tf.Rank>} 変換後のTensor
 */
export const tensorFromPixel = (
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  imgsz: [number, number]
): tf.Tensor<tf.Rank> => {
  let imageTensor = tf.browser
    .fromPixels(image)
    .toFloat()
    .div(tf.scalar(255.0));
  imageTensor = imageTensor.resizeBilinear(imgsz);
  return (imageTensor = imageTensor.expandDims(0));
};

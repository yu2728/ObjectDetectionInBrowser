import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";

import { useEffect, useRef, useState } from "react";
import { convertImageElement, tensorFromPixel } from "./libs/conver_image_element";
import { loadMetadata } from "./libs/load_metadata";
import { loadYOLOModel } from "./libs/load_model";
import { getImagePath, getModelPath } from "./libs/model_path";
import { predict } from "./libs/predict";
import { DetectBbox, ModelTaskType, OrientedBbox, SegBbox, YOLOMetadata } from "./libs/types";
import { detectView } from "./libs/view/detect";
import { orientedView } from "./libs/view/oriented";
import { segmentView } from "./libs/view/seg";

function App() {
  const [selectedModelType, setsSelectedModelType] = useState<ModelTaskType>(ModelTaskType.DETECT)
  const [model, setModel] = useState<tf.GraphModel<string | tf.io.IOHandler> | null>(null)
  const [metadata, setMetadata] = useState<YOLOMetadata | null>(null)

  const [state, setState] = useState<[boolean, string]>([false, ""])
  const imageRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  function handleChangeModel(event: React.ChangeEvent<HTMLSelectElement>) {
    const taskType = event.target.value as ModelTaskType
    setsSelectedModelType(taskType)
    imageRef.current!.src = getImagePath(taskType)
  }

  function detect(){
    setState([true, "推論を実行しています"])
    // 画像のTensorを取得
    const convertedCanvas = convertImageElement(
      imageRef.current!,
      metadata!.imgsz
    );
    const imageTensor = tensorFromPixel(convertedCanvas, metadata!.imgsz);
    const bboxes = predict(selectedModelType, model!, imageTensor, Object.entries(metadata!.names).length, 0.5, undefined)
    const restoreScale = Math.max(
      imageRef.current!.width / metadata!.imgsz[0],
      imageRef.current!.height / metadata!.imgsz[1]
    );
    switch(selectedModelType){
      case ModelTaskType.DETECT:
        detectView(canvasRef.current!, imageRef.current!, restoreScale, bboxes as DetectBbox[])
        break;
      case ModelTaskType.ORIENTED:
        orientedView(canvasRef.current!, imageRef.current!, restoreScale, bboxes as OrientedBbox[])
        break
      case ModelTaskType.SEGMENT:
        segmentView(canvasRef.current!, imageRef.current!, restoreScale, bboxes as SegBbox[])
        break
    }
    setState([false, "完了"])

  }

  useEffect(() => {
    const fetchData = async () => {
      const path = getModelPath(selectedModelType)
      // メタデータの読み込み
      setState([true, "メタデータを読み込んでいます"])
      const metadata = await loadMetadata(path);
      if(metadata === null){
        setState([false, "メタデータの読み込みに失敗しました"])
        return
      }
      setMetadata(metadata)
      
      // modelの読み込み
      setState([true, "モデルを読み込んでいます"])
      const model = await loadYOLOModel(path, metadata.imgsz)
      if(model === null){        
        setState([false, "モデルの読み込みに失敗しました"])
        return
      }
      setModel(model)
      setState([false, "モデルの読み込みが完了しました"])
    }
    fetchData()
  }, [selectedModelType])

  return (
    <>
      <h1 className="">Object Detection in Browser</h1>
      <p>Tensorflow.jsとYOLOのモデルを使用して物体検出を行うデモサイトです。</p>
      <p>Detection, Segmantation, Oriented Bounding Boxに対応しています</p>
      <select onChange={handleChangeModel}>
        <option value={ModelTaskType.DETECT}>Detection</option>
        <option value={ModelTaskType.ORIENTED}>Oriented Bounding Box</option>
        <option value={ModelTaskType.SEGMENT}>Segmentation</option>
      </select>
      <p>state: {state[1]}</p>
      <button onClick={detect}>detect</button>
      <img ref={imageRef} src="./images/seg.jpg" alt="sample" />
      <canvas ref={canvasRef}></canvas>
    </>
  )
}

export default App

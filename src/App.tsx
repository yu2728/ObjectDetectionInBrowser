import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";

import * as tf from "@tensorflow/tfjs";

import { LoaderCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { detect } from "./libs/detect";
import { detectV12 } from "./libs/detect_v12";
import { loadMetadata } from "./libs/load_metadata";
import { loadYOLOModel } from "./libs/load_model";
import { getImagePath, getModelPath } from "./libs/model_path";
import { obb } from "./libs/obb";
import { pose } from "./libs/pose";
import { seg } from "./libs/seg";
import { DetectBbox, ModelTaskType, OrientedBbox, PoseBbox, SegBbox, YOLOMetadata } from "./libs/types";
import { detectView } from "./libs/view/detect";
import { orientedView } from "./libs/view/oriented";
import { poseView } from "./libs/view/pose";
import { segmentView } from "./libs/view/seg";

function App() {
  const [selectedModelType, setsSelectedModelType] = useState<ModelTaskType>(ModelTaskType.DETECT);
  const [model, setModel] = useState<tf.GraphModel<string | tf.io.IOHandler> | null>(null);
  const [metadata, setMetadata] = useState<YOLOMetadata | null>(null);

  const [state, setState] = useState<[boolean, string]>([false, ""]);
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageSrc, setImageSrc] = useState<string>("./images/detect.jpg");
  const inputRef = useRef<HTMLInputElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);

  function handleChangeModel(event: React.ChangeEvent<HTMLSelectElement>) {
    const taskType = event.target.value as ModelTaskType;
    setsSelectedModelType(taskType);
    imageRef.current!.src = getImagePath(taskType);
  }

  async function predict() {
    setState([true, "推論を実行しています"]);
    if (!imageRef.current || !metadata || !model) {
      return;
    }
    const restoreScale = Math.max(imageRef.current!.width / metadata!.imgsz[0], imageRef.current!.height / metadata!.imgsz[1]);
    switch (selectedModelType) {
      case ModelTaskType.DETECT: {
        const bboxes = await detect(model, imageRef.current, metadata.imgsz, 0.4);
        detectView(canvasRef.current!, imageRef.current!, restoreScale, bboxes as DetectBbox[]);
        break;
      }
      case ModelTaskType.ORIENTED: {
        const bboxes = await obb(model, imageRef.current, metadata.imgsz, 0.4);
        orientedView(canvasRef.current!, imageRef.current!, restoreScale, bboxes as OrientedBbox[]);
        break;
      }
      case ModelTaskType.SEGMENT: {
        const bboxes = await seg(model, imageRef.current, metadata.imgsz, 0.4);
        segmentView(canvasRef.current!, maskCanvasRef.current!, imageRef.current!, metadata!.imgsz, bboxes as SegBbox[]);
        break;
      }
      case ModelTaskType.POSE: {
        const bboxes = await pose(model, imageRef.current, metadata.imgsz, 0.4);
        poseView(canvasRef.current!, imageRef.current!, restoreScale, bboxes as PoseBbox[]);
        break;
      }
      case ModelTaskType.V12_DETECT: {
        const bboxes = await detectV12(model, imageRef.current, metadata.imgsz, 0.4);
        detectView(canvasRef.current!, imageRef.current!, restoreScale, bboxes);
        break;
      }
    }
    setState([false, `完了`]);
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImageSrc(reader.result as string); // 画像を表示
      };
      reader.readAsDataURL(file);
    }
  };
  useEffect(() => {
    const fetchData = async () => {
      const path = getModelPath(selectedModelType);
      if (inputRef.current) {
        inputRef.current.value = "";
      }
      // メタデータの読み込み
      setState([true, "メタデータを読み込んでいます"]);
      const metadata = await loadMetadata(path);
      if (metadata === null) {
        setState([false, "メタデータの読み込みに失敗しました"]);
        return;
      }
      setMetadata(metadata);

      // modelの読み込み
      setState([true, "モデルを読み込んでいます"]);
      const model = await loadYOLOModel(path, metadata.imgsz);
      if (model === null) {
        setState([false, "モデルの読み込みに失敗しました"]);
        return;
      }
      setModel(model);
      setState([false, "モデルの読み込みが完了しました"]);
    };
    fetchData();
  }, [selectedModelType]);

  return (
    <div className=" flex flex-col gap-4 items-center py-12">
      <h1 className=" text-3xl sm:text-4xl md:text-5xl text-center py-12 text-purple-800">Object Detection in Browser</h1>
      <p className="text-center opacity-80">
        Tensorflow.jsとYOLOのモデルを使用して物体検出を行うデモサイトです。
        <br />
        Detection, Oriented Bounding Box, Segmantationに対応しています
      </p>

      <div>
        <label htmlFor="detect_type" className="block mb-2 text-sm font-medium text-center">
          Detect Type
        </label>
        <select id="detect_type" onChange={handleChangeModel} className=" px-4 py-2 border-2 border-gray-400 rounded-xl ">
          <option value={ModelTaskType.DETECT}>Detection</option>
          <option value={ModelTaskType.ORIENTED}>Oriented Bounding Box</option>
          <option value={ModelTaskType.SEGMENT}>Segmentation</option>
          <option value={ModelTaskType.POSE}>POSE</option>
          <option value={ModelTaskType.V12_DETECT}>Detection v12</option>
        </select>
      </div>
      <label htmlFor="input_file" className="block mb-2 text-sm font-medium text-center">
        Input file
      </label>
      <input id="input_file" ref={inputRef} type="file" accept="image/*" onChange={handleFileChange} className="mb-4" />
      <div>
        <label htmlFor="state" className="block mb-2 text-sm font-medium text-center">
          state
        </label>
        <p className=" text-sm opacity-80">{state[1]}</p>
      </div>
      <button
        onClick={() => predict()}
        disabled={state[0]}
        className={` inline-flex justify-center items-center px-8 py-2 rounded-lg h-12 w-32 text-center text-white text-xl ${state[0] ? "bg-cyan-500 opacity-50 " : "bg-cyan-500"}`}
      >
        {state[0] ? <LoaderCircle className="  h-6 w-6 animate-spin" /> : "検出"}
      </button>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
        <div>
          <h2 className="text-lg font-semibold mb-2 text-center">Input Image</h2>
          <img ref={imageRef} src={imageSrc} alt="Sample for object detection" className="w-full max-w-[800px] h-auto rounded-lg shadow-md" />
        </div>
        <div>
          <h2 className="text-lg font-semibold mb-2 text-center ">Result</h2>
          <canvas ref={canvasRef} className="w-full max-w-[800px] h-auto rounded-lg shadow-md"></canvas>
        </div>
      </div>
      <canvas ref={maskCanvasRef} className="w-full max-w-[800px] h-auto rounded-lg shadow-md hidden"></canvas>
    </div>
  );
}

export default App;

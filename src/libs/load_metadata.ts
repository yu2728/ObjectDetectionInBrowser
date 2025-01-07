import { load } from "js-yaml";
import { YOLOMetadata } from "./types";

/**
 * メタデータを読み込む
 * @param {string} path メタデータのフォルダパス.最後にスラッシュはつけない
 * @returns {YOLOMetadata} メタデータをロードするPromiseを返す
 */
export async function loadMetadata(path: string): Promise<YOLOMetadata | null> {
  let metadata: YOLOMetadata | null = null;
  await fetch(`${path}/metadata.yaml`)
    .then((response) => response.text())
    .then((text) => load(text))
    .then((yamlData) => (metadata = yamlData as YOLOMetadata))
    .catch((error) => console.error("YAML読み込みエラー:", error));
  return metadata;
}

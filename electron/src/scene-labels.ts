import type { SceneDocument } from './contracts';
import { SCENE_CATALOG } from './SceneCatalog';
import { message, type LocalizedText } from './i18n';

/** Localize the built-in title without rewriting document data or imported names. */
export function sceneDisplayName(doc: SceneDocument): LocalizedText {
  const entry = SCENE_CATALOG.find((item) => item.id === doc.root.userData.catalogId);
  return entry && doc.name === entry.name ? message(entry.name) : doc.name;
}

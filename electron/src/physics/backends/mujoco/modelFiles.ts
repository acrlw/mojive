import type { MainModule } from '@mujoco/mujoco';
import type { ModelBundle } from '../../../resources/modelBundle';

/** Install resources in the engine's memory filesystem without per-byte Embind calls. */
export function installModelFiles(mj: MainModule, bundle: ModelBundle): { entry: string; dispose: () => void } {
  const root = `/mojive-${crypto.randomUUID()}`;
  const directories = new Set<string>();
  const files: string[] = [];
  const dispose = () => {
    for (const path of files.splice(0)) mj.FS.unlink(path);
    for (const path of [...directories].reverse()) mj.FS.rmdir(path);
    directories.clear();
  };
  try {
    mj.FS.mkdir(root);
    directories.add(root);
    for (const file of bundle.files) {
      const parts = file.path.split('/');
      let parent = root;
      for (const name of parts.slice(0, -1)) {
        parent += `/${name}`;
        if (!directories.has(parent)) {
          mj.FS.mkdir(parent);
          directories.add(parent);
        }
      }
      const path = `${root}/${file.path}`;
      mj.FS.writeFile(path, file.bytes);
      files.push(path);
    }
    return { entry: `${root}/${bundle.entry}`, dispose };
  } catch (error) {
    dispose();
    throw error;
  }
}

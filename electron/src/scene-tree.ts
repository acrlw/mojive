import type { SceneNode } from './contracts';
export function nodeCategory(node: SceneNode): string {
  return ['box', 'sphere', 'cylinder', 'capsule', 'ground', 'geom'].includes(node.kind)
    ? 'geom'
    : node.kind;
}
export function filterSceneNodes(nodes: SceneNode[], query: string, category: string) {
  const needle = query.trim().toLocaleLowerCase();
  return nodes.filter(
    (n) =>
      (category === 'all' || nodeCategory(n) === category) &&
      (!needle || n.name.toLocaleLowerCase().includes(needle)),
  );
}
